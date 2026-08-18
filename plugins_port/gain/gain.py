"""
Portable Gain — Faurge Portable Plugin
======================================

Single-file self-contained level & balance processor for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production gain plugin exactly.

Usage
-----
    import gain

    # In-memory API
    audio_out, meta = gain.process(audio, gain_db=-3.0, stereo_balance=0.5)
"""

import sys
import os
import json
import time
import typing as T

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np


# ── WAV I/O ──────────────────────────────────────────────────────────────────


def read_wav(path: str) -> T.Tuple[np.ndarray, int]:
    """Read WAV file, return (float32 array -1..1, sample_rate)."""
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.float32:
        audio = data.astype(np.float32)
    elif data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio /= peak
    return audio, sr


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 audio (-1..1) to WAV file."""
    import scipy.io.wavfile as wavfile
    audio_clip = np.clip(audio, -1.0, 1.0)
    data = (audio_clip * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, data)


# ── Config Defaults ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "gain_db":           0.0,
    "stereo_balance":    0.0,
}


# ── Metrics Helpers ──────────────────────────────────────────────────────────


def _peak_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-30:
        return -120.0
    return float(20.0 * np.log10(peak))


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-30:
        return -120.0
    return float(20.0 * np.log10(rms))


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ── LUFS Measurement (ITU-R BS.1770-4) ───────────────────────────────────────


def _measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Measure integrated LUFS (ITU-R BS.1770-4) of audio."""
    if audio.size == 0:
        return -120.0

    was_1d = audio.ndim == 1
    if was_1d:
        channels = [audio]
        n_ch = 1
    else:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]
        n_ch = audio.shape[1]

    # K-weighting filters — pre-filter + RLB
    # First-order high-pass at 129.4 Hz
    def _pre_filter(x, sr):
        w0 = 2.0 * np.pi * 129.4 / sr
        b = np.array([1.0, -1.0], dtype=np.float64)
        a = np.array([1.0, -np.exp(-w0)], dtype=np.float64)
        from scipy.signal import lfilter
        return lfilter(b, a, x)

    # Second-order RLB shelving
    def _rlb_filter(x, sr):
        f0 = 38.0
        Q = 0.5005
        w0 = 2.0 * np.pi * f0 / sr
        cos_w0 = np.cos(w0)
        alpha = np.sin(w0) / (2.0 * Q)
        G = 3.99984385397
        A = 10.0 ** (G / 40.0)

        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        from scipy.signal import lfilter
        return lfilter(b, a, x)

    # Channel weights
    ch_weights = {0: 1.0, 1: 1.0}

    # Block size: 400 ms
    block_size = int(0.4 * sr)
    if block_size < 1:
        block_size = 1

    gated_blocks = []

    for ch_idx, ch in enumerate(channels):
        weight = ch_weights.get(ch_idx, 0.0)
        if weight == 0.0:
            continue

        filtered = _pre_filter(ch, sr)
        filtered = _rlb_filter(filtered, sr)

        n_frames = len(filtered)
        for start in range(0, n_frames, block_size):
            end = min(start + block_size, n_frames)
            block = filtered[start:end]
            if len(block) < block_size:
                continue
            mean_sq = float(np.mean(block.astype(np.float64) ** 2))
            if mean_sq > 0:
                gated_blocks.append(mean_sq * weight)

    if not gated_blocks:
        return -120.0

    # Absolute gate: -70 LUFS
    abs_thresh = 10.0 ** ((-70.0 + 0.691) / 10.0)
    after_abs = [b for b in gated_blocks if b > abs_thresh]
    if not after_abs:
        return -120.0

    # Relative gate: discard blocks below (gated_lufs - 10)
    mean_abs = float(np.mean(after_abs))
    lufs_abs = -0.691 + 10.0 * np.log10(mean_abs)
    rel_thresh = 10.0 ** ((lufs_abs - 10.0 + 0.691) / 10.0)
    after_rel = [b for b in after_abs if b > rel_thresh]
    if not after_rel:
        return -120.0

    mean_rel = float(np.mean(after_rel))
    return float(-0.691 + 10.0 * np.log10(mean_rel))


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int = None, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through level & balance.

    Args:
        audio: Input float32 array, shape (N,) for mono or (N, C) for multi-channel.
        sr: Sample rate in Hz (required for LUFS measurement).
        **config: Override any parameter from DEFAULT_CONFIG.

    Returns:
        (processed_audio, metadata_dict)
    """
    t0 = time.time()

    cfg = dict(DEFAULT_CONFIG)
    for k, v in config.items():
        k_clean = k.lower().replace("-", "_")
        if k_clean in cfg:
            cfg[k_clean] = float(v)

    audio = np.asarray(audio, dtype=np.float64)
    was_1d = audio.ndim == 1
    n_samples = audio.shape[0]

    if n_samples == 0:
        return audio.astype(np.float32), {
            "success": False,
            "processing_time_ms": 0.0,
            "input_peak_db": -120.0,
            "output_peak_db": -120.0,
            "input_rms_db": -120.0,
            "output_rms_db": -120.0,
            "input_lufs": -120.0,
            "output_lufs": -120.0,
            "peak_change_db": 0.0,
            "rms_change_db": 0.0,
            "applied_balance": 0.0,
            "clipping": False,
            "frames_processed": 0,
        }

    input_peak = _peak_db(audio)
    input_rms = _rms_db(audio)
    input_lufs = _measure_lufs(audio, sr or 44100)

    gain_lin = _db_to_linear(cfg["gain_db"])
    balance = float(cfg["stereo_balance"])

    out = np.array(audio, dtype=np.float64)

    is_stereo = not was_1d and audio.shape[1] == 2

    if is_stereo:
        if balance <= 0.0:
            left_gain = 1.0
            right_gain = 1.0 + balance
        else:
            left_gain = 1.0 - balance
            right_gain = 1.0
        left_gain *= gain_lin
        right_gain *= gain_lin
        out[:, 0] *= left_gain
        out[:, 1] *= right_gain
    else:
        out *= gain_lin

    output_peak = _peak_db(out)
    output_rms = _rms_db(out)
    output_lufs = _measure_lufs(out, sr or 44100)

    elapsed_ms = (time.time() - t0) * 1000.0

    return out.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": output_peak,
        "input_rms_db": input_rms,
        "output_rms_db": output_rms,
        "input_lufs": input_lufs,
        "output_lufs": output_lufs,
        "peak_change_db": output_peak - input_peak,
        "rms_change_db": output_rms - input_rms,
        "applied_balance": balance,
        "clipping": output_peak >= 0.0,
        "frames_processed": n_samples,
    }


def process_file(
    input_path: str,
    output_path: str = None,
    verbose: bool = False,
    **config
) -> dict:
    """Read WAV → process → write WAV."""
    audio, sr = read_wav(input_path)

    if verbose:
        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        print(f"[gain] Input: {input_path}", file=sys.stderr)
        print(f"[gain]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[gain]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[gain]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[gain] Output written: {output_path}", file=sys.stderr)

    return result


def quality_report(original: np.ndarray, processed: np.ndarray, sample_rate: int) -> dict:
    """One-shot quality assessment."""
    orig = np.asarray(original, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)

    if orig.ndim > 1:
        orig = np.mean(orig, axis=1)
    if proc.ndim > 1:
        proc = np.mean(proc, axis=1)

    min_len = min(len(orig), len(proc))
    orig = orig[:min_len]
    proc = proc[:min_len]

    noise = orig - proc
    sig_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)

    snr_db = (
        float(10.0 * np.log10(sig_power / noise_power))
        if noise_power > 1e-30 else float('inf')
    )

    before_power = np.mean(orig ** 2)
    after_power = np.mean(proc ** 2)
    improvement_db = (
        float(10.0 * np.log10(after_power / before_power))
        if before_power > 1e-30 else 0.0
    )

    return {
        "snr_db": snr_db,
        "peak_before": float(np.max(np.abs(orig))),
        "peak_after": float(np.max(np.abs(proc))),
        "rms_before": float(np.sqrt(np.mean(orig ** 2))),
        "rms_after": float(np.sqrt(np.mean(proc ** 2))),
        "improvement_db": improvement_db,
    }


# ── CLI Entry Point ─────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Portable Gain — Level & Balance")
    sub = parser.add_subparsers(dest="command")

    gain_cmd = sub.add_parser("gain", help="Adjust gain")
    gain_cmd.add_argument("input", type=str)
    gain_cmd.add_argument("output", type=str)
    gain_cmd.add_argument("--gain", type=float, default=0.0)
    gain_cmd.add_argument("--balance", type=float, default=0.0)
    gain_cmd.add_argument("--json", action="store_true")
    gain_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "gain":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            gain_db=args.gain,
            stereo_balance=args.balance,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE GAIN — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Input RMS:        {result.get('input_rms_db', 0):.1f} dB")
            print(f"  Output RMS:       {result.get('output_rms_db', 0):.1f} dB")
            print(f"  Input LUFS:       {result.get('input_lufs', 0):.2f}")
            print(f"  Output LUFS:      {result.get('output_lufs', 0):.2f}")
            print(f"  Balance:          {result.get('applied_balance', 0):+.2f}")
            print(f"  Clipping:         {result.get('clipping', False)}")
            print(f"  Frames processed: {result.get('frames_processed', 0)}")
            print(f"==========================================\n")
    elif args.command == "quality":
        orig, sr1 = read_wav(args.original)
        proc, sr2 = read_wav(args.processed)
        report = quality_report(orig, proc, sr1)
        print(json.dumps(report, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
