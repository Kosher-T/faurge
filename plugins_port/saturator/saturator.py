"""
Portable Saturator — Faurge Portable Plugin
============================================

Single-file self-contained harmonic saturator for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production saturator exactly.

Usage
-----
    import saturator

    # In-memory API
    audio_out, meta = saturator.process(audio, sr, drive_db=6.0, sat_type="tube")
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
    "drive_db":       0.0,
    "sat_type":       "tube",
    "hpf_hz":         20.0,
    "lpf_hz":         20000.0,
    "mix":            1.0,
    "oversampling":   1,
    "output_trim_db": 0.0,
}

SAT_TYPE_MAP = {
    0: "tube",
    "tube": "tube",
    1: "tape",
    "tape": "tape",
    2: "diode",
    "diode": "diode",
    3: "asymmetric",
    "asymmetric": "asymmetric",
}


# ── Waveshaper Functions ─────────────────────────────────────────────────────


def _waveshaper_tube(x: np.ndarray) -> np.ndarray:
    g = 1.3 * x
    return g / (1.0 + np.abs(g))


def _waveshaper_tape(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _waveshaper_diode(x: np.ndarray) -> np.ndarray:
    out = np.where(x >= 0, 1.0 - np.exp(-x), np.exp(x) - 1.0)
    return out


def _waveshaper_asymmetric(x: np.ndarray) -> np.ndarray:
    g = np.where(x >= 0, 1.5 * x, 0.7 * x)
    return g / (1.0 + np.abs(g))


WAVESHAPERS = {
    "tube": _waveshaper_tube,
    "tape": _waveshaper_tape,
    "diode": _waveshaper_diode,
    "asymmetric": _waveshaper_asymmetric,
}


# ── Biquad Helpers (Butterworth LPF/HPF) ────────────────────────────────────


BUTTER_Q1 = 0.5412
BUTTER_Q2 = 1.3066


def _design_lp(freq: float, sr: int, q: float = 0.7071) -> T.Tuple[np.ndarray, np.ndarray]:
    w0 = 2.0 * np.pi * freq / float(sr)
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0]),
        np.array([1.0, a1 / a0, a2 / a0])
    )


def _design_hp(freq: float, sr: int, q: float = 0.7071) -> T.Tuple[np.ndarray, np.ndarray]:
    w0 = 2.0 * np.pi * freq / float(sr)
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0]),
        np.array([1.0, a1 / a0, a2 / a0])
    )


def _apply_biquad(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter
    return lfilter(b, a, x)


def _apply_butterworth_lp(x: np.ndarray, freq: float, sr: int) -> np.ndarray:
    b1, a1 = _design_lp(freq, sr, BUTTER_Q1)
    b2, a2 = _design_lp(freq, sr, BUTTER_Q2)
    y = _apply_biquad(b1, a1, x)
    y = _apply_biquad(b2, a2, y)
    return y


def _apply_butterworth_hp(x: np.ndarray, freq: float, sr: int) -> np.ndarray:
    b1, a1 = _design_hp(freq, sr, BUTTER_Q1)
    b2, a2 = _design_hp(freq, sr, BUTTER_Q2)
    y = _apply_biquad(b1, a1, x)
    y = _apply_biquad(b2, a2, y)
    return y


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


def _process_mono(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process a mono channel through the saturator."""
    n = len(audio)
    dry = np.array(audio, dtype=np.float64)
    wet = np.array(audio, dtype=np.float64)

    sat_type = SAT_TYPE_MAP.get(cfg["sat_type"], "tube")
    drive_lin = _db_to_linear(cfg["drive_db"])
    trim_lin = _db_to_linear(cfg["output_trim_db"])
    oversampling = max(1, min(4, int(cfg["oversampling"])))
    ws = WAVESHAPERS[sat_type]

    # HPF (pre-distortion)
    if cfg["hpf_hz"] > 20.0:
        wet = _apply_butterworth_hp(wet, cfg["hpf_hz"], sr)

    # Drive
    wet *= drive_lin

    # Waveshaping with oversampling
    if oversampling > 1 and n > 0:
        up = oversampling
        up_n = n * up
        up_t = np.arange(up_n) / (sr * up)
        up_idx = np.arange(up_n) // up
        up_idx = np.clip(up_idx, 0, n - 1)

        up_wet = np.zeros(up_n, dtype=np.float64)
        up_wet[::up] = wet

        aa_freq = float(sr) / 2.0
        up_wet = _apply_butterworth_lp(up_wet, aa_freq, sr * up)

        up_wet = ws(up_wet)

        up_wet = _apply_butterworth_lp(up_wet, aa_freq, sr * up)

        wet = up_wet[::up]
        if len(wet) > n:
            wet = wet[:n]
        elif len(wet) < n:
            wet = np.pad(wet, (0, n - len(wet)))
    else:
        wet = ws(wet)

    # LPF (post-distortion)
    if cfg["lpf_hz"] < 20000.0:
        wet = _apply_butterworth_lp(wet, cfg["lpf_hz"], sr)

    # Harmonic distortion measurement
    harm_sum = float(np.sum(np.abs(wet - dry[:len(wet)])))

    # Mix and trim
    out = wet * cfg["mix"] + dry[:len(wet)] * (1.0 - cfg["mix"])
    out *= trim_lin

    dc_offset = float(np.mean(out))

    return out, {
        "avg_harmonic_db": _rms_db(np.array([harm_sum / max(n, 1)])),
        "dc_offset": dc_offset,
    }


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the harmonic saturator.

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
            if k_clean == "sat_type":
                cfg[k_clean] = SAT_TYPE_MAP.get(v, "tube")
            elif k_clean == "oversampling":
                cfg[k_clean] = int(v)
            else:
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
            "avg_harmonic_db": -120.0,
            "dc_offset": 0.0,
            "frames_processed": 0,
        }

    input_peak = _peak_db(audio)
    input_rms = _rms_db(audio)

    if was_1d:
        processed, sat_info = _process_mono(audio, sr, cfg)
        processed_audio = processed
    else:
        n_ch = audio.shape[1]
        outputs = []
        sat_info = {"avg_harmonic_db": -120.0, "dc_offset": 0.0}
        for ch in range(n_ch):
            ch_out, ch_info = _process_mono(audio[:, ch], sr, cfg)
            outputs.append(ch_out)
            if ch_info["avg_harmonic_db"] > sat_info["avg_harmonic_db"]:
                sat_info["avg_harmonic_db"] = ch_info["avg_harmonic_db"]
            sat_info["dc_offset"] = max(sat_info["dc_offset"], abs(ch_info["dc_offset"]))
        processed_audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed_audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": _peak_db(processed_audio),
        "input_rms_db": input_rms,
        "output_rms_db": _rms_db(processed_audio),
        "avg_harmonic_db": sat_info["avg_harmonic_db"],
        "dc_offset": sat_info["dc_offset"],
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
        print(f"[saturator] Input: {input_path}", file=sys.stderr)
        print(f"[saturator]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[saturator]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[saturator]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[saturator] Output written: {output_path}", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Portable Saturator — Harmonic Saturator")
    sub = parser.add_subparsers(dest="command")

    sat_cmd = sub.add_parser("saturate", help="Saturate audio")
    sat_cmd.add_argument("input", type=str)
    sat_cmd.add_argument("output", type=str)
    sat_cmd.add_argument("--drive", type=float, default=0.0)
    sat_cmd.add_argument("--mix", type=float, default=1.0)
    sat_cmd.add_argument("--type", type=str, default="tube",
                         choices=["tube", "tape", "diode", "asymmetric"])
    sat_cmd.add_argument("--hpf", type=float, default=20.0)
    sat_cmd.add_argument("--lpf", type=float, default=20000.0)
    sat_cmd.add_argument("--oversampling", type=int, default=1, choices=[1, 2, 4])
    sat_cmd.add_argument("--output-trim", type=float, default=0.0)
    sat_cmd.add_argument("--json", action="store_true")
    sat_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "saturate":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            drive_db=args.drive,
            mix=args.mix,
            sat_type=args.type,
            hpf_hz=args.hpf,
            lpf_hz=args.lpf,
            oversampling=args.oversampling,
            output_trim_db=args.output_trim,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE SATURATOR — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Avg harmonic:     {result.get('avg_harmonic_db', 0):.1f} dB")
            print(f"  DC offset:        {result.get('dc_offset', 0):.6f}")
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
