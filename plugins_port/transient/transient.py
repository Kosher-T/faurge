"""
Portable Transient — Faurge Portable Plugin
=============================================

Single-file self-contained transient shaper for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production transient shaper exactly.

Usage
-----
    import transient

    # In-memory API
    audio_out, meta = transient.process(audio, sr, attack_gain_db=12.0, sustain_gain_db=-6.0)
"""

import sys
import os
import json
import time
import typing as T

# Ensure parent of plugins_port is on sys.path for Kaggle imports
_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


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
    "attack_gain_db":     6.0,
    "sustain_gain_db":    0.0,
    "attack_time_ms":     5.0,
    "release_time_ms":    100.0,
    "sensitivity_db":     -30.0,
    "mix":                1.0,
}


# ── RC Alpha ─────────────────────────────────────────────────────────────────


def _rc_alpha(tau_ms: float, sr: int) -> float:
    if tau_ms <= 0.0 or sr <= 0:
        return 1.0
    samples = tau_ms * float(sr) * 0.001
    return 1.0 - np.exp(-2.2 / samples)


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


# ── Processing Core ──────────────────────────────────────────────────────────


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _process_mono_jit(out, n, attack_alpha, release_alpha, attack_lin, sustain_lin, sens_lin, mix):
        envelope = 0.0
        sum_attack_gain = 0.0
        sum_sustain_gain = 0.0
        attack_frames = 0
        sustain_frames = 0

        for i in range(n):
            x = abs(out[i])

            if x > envelope:
                alpha = attack_alpha
            else:
                alpha = release_alpha
            envelope += alpha * (x - envelope)

            max_val = max(x, envelope)
            if max_val < sens_lin:
                factor = 0.0
            else:
                diff = x - envelope
                if diff > 0:
                    factor = diff / max(max_val, 1e-30)
                else:
                    factor = -(-diff) / max(max_val, 1e-30)

            if factor > 0:
                gain = 1.0 + (attack_lin - 1.0) * factor
                sum_attack_gain += gain
                attack_frames += 1
            elif factor < 0:
                gain = 1.0 + (sustain_lin - 1.0) * (-factor)
                sum_sustain_gain += gain
                sustain_frames += 1
            else:
                gain = 1.0

            wet = out[i] * gain
            out[i] = wet * mix + out[i] * (1.0 - mix)

        return sum_attack_gain, sum_sustain_gain, attack_frames, sustain_frames


def _process_mono(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process a mono channel through the transient shaper."""
    n = len(audio)
    out = np.array(audio, dtype=np.float64)

    attack_lin = _db_to_linear(cfg["attack_gain_db"])
    sustain_lin = _db_to_linear(cfg["sustain_gain_db"])
    sens_lin = _db_to_linear(cfg["sensitivity_db"])
    attack_alpha = _rc_alpha(cfg["attack_time_ms"], sr)
    release_alpha = _rc_alpha(cfg["release_time_ms"], sr)
    mix = cfg["mix"]

    if NUMBA_AVAILABLE:
        out = np.array(audio, dtype=np.float64)
        sum_attack_gain, sum_sustain_gain, attack_frames, sustain_frames = \
            _process_mono_jit(out, n, attack_alpha, release_alpha, attack_lin, sustain_lin, sens_lin, mix)
    else:
        out = np.array(audio, dtype=np.float64)
        envelope = 0.0
        sum_attack_gain = 0.0
        sum_sustain_gain = 0.0
        attack_frames = 0
        sustain_frames = 0

        for i in range(n):
            x = abs(out[i])

            if x > envelope:
                alpha = attack_alpha
            else:
                alpha = release_alpha
            envelope += alpha * (x - envelope)

            max_val = max(x, envelope)
            if max_val < sens_lin:
                factor = 0.0
            else:
                diff = x - envelope
                if diff > 0:
                    factor = diff / max(max_val, 1e-30)
                else:
                    factor = -(-diff) / max(max_val, 1e-30)

            if factor > 0:
                gain = 1.0 + (attack_lin - 1.0) * factor
                sum_attack_gain += gain
                attack_frames += 1
            elif factor < 0:
                gain = 1.0 + (sustain_lin - 1.0) * (-factor)
                sum_sustain_gain += gain
                sustain_frames += 1
            else:
                gain = 1.0

            wet = out[i] * gain
            out[i] = wet * mix + out[i] * (1.0 - mix)

    peak_val = float(np.max(np.abs(out))) if n > 0 else 0.0
    rms_val = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) if n > 0 else 1e-30
    peak_to_rms = _linear_to_db(peak_val / max(rms_val, 1e-30)) if rms_val > 0 else 0.0

    avg_attack_db = _linear_to_db(sum_attack_gain / max(attack_frames, 1)) if attack_frames > 0 else 0.0
    avg_sustain_db = _linear_to_db(sum_sustain_gain / max(sustain_frames, 1)) if sustain_frames > 0 else 0.0

    return out, {
        "peak_to_rms_db": peak_to_rms,
        "avg_attack_db": avg_attack_db,
        "avg_sustain_db": avg_sustain_db,
    }


def _linear_to_db(linear: float) -> float:
    if linear < 1e-30:
        return -120.0
    return 20.0 * np.log10(linear)


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the transient shaper.

    Args:
        audio: Input float32 array, shape (N,) for mono or (N, C) for multi-channel.
        sr: Sample rate in Hz.
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
            "peak_to_rms_db": 0.0,
            "avg_attack_db": 0.0,
            "avg_sustain_db": 0.0,
            "frames_processed": 0,
        }

    input_peak = _peak_db(audio)
    input_rms = _rms_db(audio)

    if was_1d:
        processed, tr_info = _process_mono(audio, sr, cfg)
        processed_audio = processed
    else:
        n_ch = audio.shape[1]
        outputs = []
        tr_info = {"peak_to_rms_db": 0.0, "avg_attack_db": 0.0, "avg_sustain_db": 0.0}
        for ch in range(n_ch):
            ch_out, ch_info = _process_mono(audio[:, ch], sr, cfg)
            outputs.append(ch_out)
            if ch_info["peak_to_rms_db"] > tr_info["peak_to_rms_db"]:
                tr_info["peak_to_rms_db"] = ch_info["peak_to_rms_db"]
        processed_audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed_audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": _peak_db(processed_audio),
        "input_rms_db": input_rms,
        "output_rms_db": _rms_db(processed_audio),
        "peak_to_rms_db": tr_info["peak_to_rms_db"],
        "avg_attack_db": tr_info["avg_attack_db"],
        "avg_sustain_db": tr_info["avg_sustain_db"],
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
        print(f"[transient] Input: {input_path}", file=sys.stderr)
        print(f"[transient]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[transient]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[transient]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[transient] Output written: {output_path}", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Portable Transient — Transient Shaper")
    sub = parser.add_subparsers(dest="command")

    tr_cmd = sub.add_parser("transient", help="Shape transients")
    tr_cmd.add_argument("input", type=str)
    tr_cmd.add_argument("output", type=str)
    tr_cmd.add_argument("--attack-gain", type=float, default=6.0)
    tr_cmd.add_argument("--sustain-gain", type=float, default=0.0)
    tr_cmd.add_argument("--attack-time", type=float, default=5.0)
    tr_cmd.add_argument("--release-time", type=float, default=100.0)
    tr_cmd.add_argument("--sensitivity", type=float, default=-30.0)
    tr_cmd.add_argument("--mix", type=float, default=1.0)
    tr_cmd.add_argument("--json", action="store_true")
    tr_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "transient":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            attack_gain_db=args.attack_gain,
            sustain_gain_db=args.sustain_gain,
            attack_time_ms=args.attack_time,
            release_time_ms=args.release_time,
            sensitivity_db=args.sensitivity,
            mix=args.mix,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE TRANSIENT — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Peak-to-RMS:      {result.get('peak_to_rms_db', 0):.1f} dB")
            print(f"  Avg attack:       {result.get('avg_attack_db', 0):.1f} dB")
            print(f"  Avg sustain:      {result.get('avg_sustain_db', 0):.1f} dB")
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
