"""
Portable Esser — Faurge Portable Plugin
========================================

Single-file self-contained dynamic sibilance processor for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production esser exactly.

Usage
-----
    import esser

    # In-memory API
    audio_out, meta = esser.process(audio, sr, center_freq_hz=6000.0, threshold_db=-30.0)
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
    "center_freq_hz": 6000.0,
    "threshold_db":   -30.0,
    "ratio":          5.0,
    "bandwidth_hz":   1500.0,
    "attack_ms":      2.0,
    "release_ms":     100.0,
}


# ── Bandpass Biquad (sibilance detector) ─────────────────────────────────────


def _design_bandpass(freq: float, q: float, sr: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """Design bandpass biquad matching C++ esser biquad."""
    w0 = 2.0 * np.pi * freq / float(sr)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0]),
        np.array([1.0, a1 / a0, a2 / a0])
    )


# ── RC Alpha ─────────────────────────────────────────────────────────────────


def _rc_alpha(tau_ms: float, sr: int) -> float:
    if tau_ms <= 0.0:
        return 1.0
    return 1.0 - np.exp(-2.2 / (tau_ms * float(sr) * 0.001))


# ── Gain Reduction ───────────────────────────────────────────────────────────


def _compute_gain_reduction(env_db: float, threshold_db: float, ratio: float) -> float:
    """Compute gain reduction matching C++ exactly."""
    overshoot = env_db - threshold_db

    if ratio >= 1.0:
        if overshoot <= 0.0:
            return 0.0
        return -overshoot * (ratio - 1.0) / ratio
    else:
        if overshoot >= 0.0:
            return 0.0
        return overshoot * (1.0 - ratio)


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


def _linear_to_db(linear: float) -> float:
    if linear < 1e-30:
        return -120.0
    return 20.0 * np.log10(linear)


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ── Processing Core ──────────────────────────────────────────────────────────


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _process_mono_jit(out, n, bp_b, bp_a, attack_alpha, release_alpha,
                          attack_alpha_gr, release_alpha_gr, threshold_db, ratio, sr):
        bp_state = np.zeros(4, dtype=np.float64)

        # Warmup (matching C++ 5ms warmup — inline biquad)
        warmup_samples = max(int(sr * 0.005), 1)
        for _ in range(warmup_samples):
            y = bp_b[0] * 0.0 + bp_b[1] * bp_state[0] + bp_b[2] * bp_state[1] \
                - bp_a[1] * bp_state[2] - bp_a[2] * bp_state[3]
            bp_state[1] = bp_state[0]
            bp_state[0] = 0.0
            bp_state[3] = bp_state[2]
            bp_state[2] = y

        envelope = 0.0
        smoothed_gain_db = 0.0
        max_gr = 0.0
        sum_active_gr = 0.0
        active_frames = 0

        for i in range(n):
            input_sample = out[i]

            # Bandpass detection (inlined biquad)
            detected = bp_b[0] * input_sample + bp_b[1] * bp_state[0] + bp_b[2] * bp_state[1] \
                       - bp_a[1] * bp_state[2] - bp_a[2] * bp_state[3]
            bp_state[1] = bp_state[0]
            bp_state[0] = input_sample
            bp_state[3] = bp_state[2]
            bp_state[2] = detected

            # Envelope follower
            env_in = np.sqrt(max(1e-30, detected * detected))
            alpha = attack_alpha if env_in >= envelope else release_alpha
            envelope += alpha * (env_in - envelope)

            # Gain reduction (inlined)
            eps = 1e-30
            env_db = 20.0 * np.log10(max(envelope, eps))
            overshoot = env_db - threshold_db
            if ratio >= 1.0:
                if overshoot <= 0.0:
                    target_gr = 0.0
                else:
                    target_gr = -overshoot * (ratio - 1.0) / ratio
            else:
                if overshoot >= 0.0:
                    target_gr = 0.0
                else:
                    target_gr = overshoot * (1.0 - ratio)

            alpha_gr = attack_alpha_gr if target_gr < smoothed_gain_db else release_alpha_gr
            smoothed_gain_db += alpha_gr * (target_gr - smoothed_gain_db)

            gain_lin = 10.0 ** (smoothed_gain_db / 20.0)
            out[i] = input_sample * gain_lin

            abs_gr = abs(smoothed_gain_db)
            if abs_gr > max_gr:
                max_gr = abs_gr
            if abs_gr > 0.1:
                sum_active_gr += abs_gr
                active_frames += 1

        return max_gr, sum_active_gr, active_frames


def _process_mono(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process a mono channel through the esser, sample-by-sample."""
    n = len(audio)
    out = np.array(audio, dtype=np.float64)

    # Bandpass (sibilance detector)
    center = max(20.0, min(cfg["center_freq_hz"], 20000.0))
    bw = max(cfg["bandwidth_hz"], 50.0)
    q = center / bw

    bp_b, bp_a = _design_bandpass(center, q, sr)

    # Effective attack: at least 2 cycles of bandpass center
    effective_attack_ms = max(cfg["attack_ms"], 2.0 * q / center * 1000.0)

    attack_alpha = _rc_alpha(effective_attack_ms, sr)
    release_alpha = _rc_alpha(cfg["release_ms"], sr)
    attack_alpha_gr = _rc_alpha(effective_attack_ms, sr)
    release_alpha_gr = _rc_alpha(cfg["release_ms"], sr)

    if NUMBA_AVAILABLE:
        out = np.array(audio, dtype=np.float64)
        max_gr, sum_active_gr, active_frames = _process_mono_jit(
            out, n, bp_b, bp_a, attack_alpha, release_alpha,
            attack_alpha_gr, release_alpha_gr,
            cfg["threshold_db"], cfg["ratio"], sr,
        )
    else:
        out = np.array(audio, dtype=np.float64)

        # Bandpass state (x1, x2, y1, y2)
        bp_state = [0.0, 0.0, 0.0, 0.0]

        # Warmup (matching C++ 5ms warmup)
        warmup_samples = max(int(sr * 0.005), 1)
        for _ in range(warmup_samples):
            y = bp_b[0] * 0.0 + bp_b[1] * bp_state[0] + bp_b[2] * bp_state[1] \
                - bp_a[1] * bp_state[2] - bp_a[2] * bp_state[3]
            bp_state[1] = bp_state[0]
            bp_state[0] = 0.0
            bp_state[3] = bp_state[2]
            bp_state[2] = y

        envelope = 0.0
        smoothed_gain_db = 0.0
        max_gr = 0.0
        sum_active_gr = 0.0
        active_frames = 0

        for i in range(n):
            input_sample = out[i]

            # Bandpass detection
            detected = bp_b[0] * input_sample + bp_b[1] * bp_state[0] + bp_b[2] * bp_state[1] \
                       - bp_a[1] * bp_state[2] - bp_a[2] * bp_state[3]
            bp_state[1] = bp_state[0]
            bp_state[0] = input_sample
            bp_state[3] = bp_state[2]
            bp_state[2] = detected

            # Envelope follower
            env_in = np.sqrt(max(1e-30, detected * detected))
            alpha = attack_alpha if env_in >= envelope else release_alpha
            envelope += alpha * (env_in - envelope)

            # Gain reduction
            env_db = _linear_to_db(envelope)
            target_gr = _compute_gain_reduction(env_db, cfg["threshold_db"], cfg["ratio"])

            alpha_gr = attack_alpha_gr if target_gr < smoothed_gain_db else release_alpha_gr
            smoothed_gain_db += alpha_gr * (target_gr - smoothed_gain_db)

            gain_lin = _db_to_linear(smoothed_gain_db)
            out[i] = input_sample * gain_lin

            abs_gr = abs(smoothed_gain_db)
            if abs_gr > max_gr:
                max_gr = abs_gr
            if abs_gr > 0.1:
                sum_active_gr += abs_gr
                active_frames += 1

    return out, {
        "max_gain_reduction_db": max_gr,
        "avg_active_gain_reduction_db": sum_active_gr / max(active_frames, 1),
        "sibilant_frames": active_frames,
    }


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the de-esser.

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
            "max_gain_reduction_db": 0.0,
            "avg_active_gain_reduction_db": 0.0,
            "sibilant_frames": 0,
            "frames_processed": 0,
        }

    input_peak = _peak_db(audio)
    input_rms = _rms_db(audio)

    if was_1d:
        processed, gr_info = _process_mono(audio, sr, cfg)
        processed_audio = processed
    else:
        n_ch = audio.shape[1]
        outputs = []
        gr_info = {"max_gain_reduction_db": 0.0, "avg_active_gain_reduction_db": 0.0, "sibilant_frames": 0}
        for ch in range(n_ch):
            ch_out, ch_gr = _process_mono(audio[:, ch], sr, cfg)
            outputs.append(ch_out)
            if ch_gr["max_gain_reduction_db"] > gr_info["max_gain_reduction_db"]:
                gr_info["max_gain_reduction_db"] = ch_gr["max_gain_reduction_db"]
            gr_info["sibilant_frames"] += ch_gr["sibilant_frames"]
        processed_audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed_audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": _peak_db(processed_audio),
        "input_rms_db": input_rms,
        "output_rms_db": _rms_db(processed_audio),
        "max_gain_reduction_db": gr_info["max_gain_reduction_db"],
        "avg_active_gain_reduction_db": gr_info["avg_active_gain_reduction_db"],
        "sibilant_frames": gr_info["sibilant_frames"],
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
        print(f"[esser] Input: {input_path}", file=sys.stderr)
        print(f"[esser]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[esser]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[esser]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[esser] Output written: {output_path}", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Portable Esser — Dynamic Sibilance Processor")
    sub = parser.add_subparsers(dest="command")

    esser_cmd = sub.add_parser("esser", help="Process audio with de-esser")
    esser_cmd.add_argument("input", type=str)
    esser_cmd.add_argument("output", type=str)
    esser_cmd.add_argument("--center-freq", type=float, default=6000.0)
    esser_cmd.add_argument("--threshold", type=float, default=-30.0)
    esser_cmd.add_argument("--ratio", type=float, default=5.0)
    esser_cmd.add_argument("--bandwidth", type=float, default=1500.0)
    esser_cmd.add_argument("--attack", type=float, default=2.0)
    esser_cmd.add_argument("--release", type=float, default=100.0)
    esser_cmd.add_argument("--json", action="store_true")
    esser_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "esser":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            center_freq_hz=args.center_freq,
            threshold_db=args.threshold,
            ratio=args.ratio,
            bandwidth_hz=args.bandwidth,
            attack_ms=args.attack,
            release_ms=args.release,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE ESSER — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Max GR:           {result.get('max_gain_reduction_db', 0):.1f} dB")
            print(f"  Sibilant frames:  {result.get('sibilant_frames', 0)}")
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
