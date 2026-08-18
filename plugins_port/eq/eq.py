"""
Portable EQ — Faurge Portable Plugin
=====================================

Single-file self-contained 31-band parametric equalizer for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and parallel filter design of the C++ production EQ exactly.

Usage
-----
    import eq

    # File-based API
    result = eq.process_file("input.wav", "output.wav",
                             band1_freq=100.0, band1_gain=6.0, band1_type="low_shelf")

    # In-memory API
    audio_out, meta = eq.process(audio, sr, bands=[{"freq_hz": 1000.0, "gain_db": -3.0, "q": 1.4}])
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
from scipy.signal import lfilter


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


# ── Biquad Coefficients ──────────────────────────────────────────────────────

FILTER_TYPE_MAP = {
    0: "peak",
    "peak": "peak",
    1: "low_shelf",
    "low_shelf": "low_shelf",
    2: "high_shelf",
    "high_shelf": "high_shelf",
    3: "highpass",
    "highpass": "highpass",
    4: "lowpass",
    "lowpass": "lowpass",
    5: "bandpass",
    "bandpass": "bandpass",
    6: "notch",
    "notch": "notch"
}

DEFAULT_BAND = {
    "freq_hz": 1000.0,
    "gain_db": 0.0,
    "q": 1.0,
    "filter_type": "peak",
    "stereo_skew_db": 0.0,
    "dynamic_depth": 0.0
}


def design_biquad(filter_type: str, freq: float, gain_db: float, q: float, sr: float) -> T.Tuple[np.ndarray, np.ndarray]:
    """Design a biquad filter using standard RBJ formulas matching C++ implementation."""
    freq = float(freq)
    gain_db = float(gain_db)
    q = float(q)
    sr = float(sr)

    # Safe bounds
    freq = max(20.0, min(freq, sr / 2.0 - 1.0))
    q = max(0.1, min(q, 10.0))

    w0 = 2.0 * np.pi * freq / sr
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * q)

    if filter_type == "peak":
        A = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
    elif filter_type == "low_shelf":
        A = 10.0 ** (gain_db / 40.0)
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
    elif filter_type == "high_shelf":
        A = 10.0 ** (gain_db / 40.0)
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
    elif filter_type == "highpass":
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == "lowpass":
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == "bandpass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == "notch":
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    )


# ── Metrics Helper ───────────────────────────────────────────────────────────


def _peak_db(audio: np.ndarray) -> float:
    """Compute peak level in dBFS."""
    if audio.size == 0:
        return -120.0
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-30:
        return -120.0
    return float(20.0 * np.log10(peak))


def _rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dBFS."""
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-30:
        return -120.0
    return float(20.0 * np.log10(rms))


# ── Processing Core ──────────────────────────────────────────────────────────


def process_channel(ch_audio: np.ndarray, sr: int, bands: T.List[dict], channel_idx: int) -> np.ndarray:
    """Process a single channel through 31 parallel bands."""
    num_samples = len(ch_audio)
    output = np.zeros(num_samples, dtype=np.float64)

    for b in range(31):
        band = bands[b]
        band_gain = band["gain_db"]

        # Apply stereo skew if applicable
        if channel_idx >= 0 and band["stereo_skew_db"] != 0.0:
            half_skew = band["stereo_skew_db"] * 0.5
            band_gain += half_skew if channel_idx == 0 else -half_skew

        # Design biquad
        b_coeff, a_coeff = design_biquad(
            band["filter_type"],
            band["freq_hz"],
            band_gain,
            band["q"],
            sr
        )

        # Filter the original input (parallel configuration)
        band_buf = lfilter(b_coeff, a_coeff, ch_audio)
        output += band_buf

    # Restore dry-signal cancellation
    output -= 30.0 * ch_audio
    return output


# ── Public API ──────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, bands: T.List[dict] = None, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the 31-band parametric EQ.

    Args:
        audio: Input float32 array, shape (N,) for mono or (N, C) for multi-channel.
        sr: Sample rate in Hz.
        bands: List of up to 31 band configuration dicts.
        **config: Keyword arguments of format `bandN_param` to override or set bands.

    Returns:
        (processed_audio, metadata_dict)
    """
    t0 = time.time()

    audio = np.asarray(audio, dtype=np.float64)
    was_1d = audio.ndim == 1
    n_samples = audio.shape[0] if not was_1d else len(audio)

    if n_samples == 0:
        return audio, {
            "success": False,
            "processing_time_ms": 0.0,
            "input_peak_db": -120.0,
            "output_peak_db": -120.0,
            "input_rms_db": -120.0,
            "output_rms_db": -120.0,
            "frames_processed": 0,
        }

    # Initialize all 31 bands with defaults
    bands_list = [dict(DEFAULT_BAND) for _ in range(31)]
    if bands is not None:
        for idx, b in enumerate(bands):
            if idx >= 31:
                break
            for k, v in b.items():
                if k in DEFAULT_BAND:
                    if k == "filter_type":
                        bands_list[idx][k] = FILTER_TYPE_MAP.get(v, "peak")
                    else:
                        bands_list[idx][k] = v

    # Parse overrides from **config
    for key, val in config.items():
        key_clean = key.lower().replace("-", "_")
        if key_clean.startswith("band"):
            parts = key_clean[4:].split("_")
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                try:
                    band_num = int(parts[0])
                    param = "_".join(parts[1:])
                    if 1 <= band_num <= 31:
                        param_map = {
                            "freq": "freq_hz",
                            "freq_hz": "freq_hz",
                            "gain": "gain_db",
                            "gain_db": "gain_db",
                            "q": "q",
                            "type": "filter_type",
                            "filter_type": "filter_type",
                            "skew": "stereo_skew_db",
                            "stereo_skew_db": "stereo_skew_db",
                            "dynamic": "dynamic_depth",
                            "dynamic_depth": "dynamic_depth"
                        }
                        if param in param_map:
                            dict_key = param_map[param]
                            if dict_key == "filter_type":
                                bands_list[band_num - 1][dict_key] = FILTER_TYPE_MAP.get(val, "peak")
                            else:
                                bands_list[band_num - 1][dict_key] = float(val)
                except ValueError:
                    pass

    # Extract channels
    if was_1d:
        channels = [audio]
    else:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]

    outputs = []
    input_peak_db = -120.0
    input_rms_db = -120.0
    output_peak_db = -120.0
    output_rms_db = -120.0

    for idx, ch_audio in enumerate(channels):
        ch_out = process_channel(ch_audio, sr, bands_list, idx if not was_1d else -1)
        outputs.append(ch_out)

        if idx == 0:
            input_peak_db = _peak_db(ch_audio)
            input_rms_db = _rms_db(ch_audio)
            output_peak_db = _peak_db(ch_out)
            output_rms_db = _rms_db(ch_out)

    if was_1d:
        processed_audio = outputs[0]
    else:
        processed_audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed_audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak_db,
        "output_peak_db": output_peak_db,
        "input_rms_db": input_rms_db,
        "output_rms_db": output_rms_db,
        "frames_processed": n_samples,
    }


def process_file(
    input_path: str,
    output_path: str = None,
    bands: T.List[dict] = None,
    verbose: bool = False,
    **config
) -> dict:
    """Read WAV → process → write WAV."""
    audio, sr = read_wav(input_path)

    if verbose:
        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        print(f"[eq] Input: {input_path}", file=sys.stderr)
        print(f"[eq]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[eq]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[eq]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, bands=bands, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[eq] Output written: {output_path}", file=sys.stderr)

    return result


def quality_report(original: np.ndarray, processed: np.ndarray, sample_rate: int) -> dict:
    """One-shot quality assessment.

    Computes SNR, peak, RMS, and improvement between original and processed.
    Multi-channel is averaged to mono before comparison.
    """
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
    parser = argparse.ArgumentParser(description="Portable EQ — 31-band Parametric Equalizer")
    sub = parser.add_subparsers(dest="command")

    eq_cmd = sub.add_parser("eq", help="Process audio with EQ")
    eq_cmd.add_argument("input", type=str)
    eq_cmd.add_argument("output", type=str)

    # Dynamic CLI argument parsing for 31 bands
    for b in range(1, 32):
        eq_cmd.add_argument(f"--band{b}-freq", type=float, default=1000.0)
        eq_cmd.add_argument(f"--band{b}-gain", type=float, default=0.0)
        eq_cmd.add_argument(f"--band{b}-q", type=float, default=1.0)
        eq_cmd.add_argument(f"--band{b}-type", type=str, default="peak")
        eq_cmd.add_argument(f"--band{b}-skew", type=float, default=0.0)
        eq_cmd.add_argument(f"--band{b}-dynamic", type=float, default=0.0)

    eq_cmd.add_argument("--json", action="store_true")
    eq_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "eq":
        config = {}
        for b in range(1, 32):
            config[f"band{b}_freq"] = getattr(args, f"band{b}_freq")
            config[f"band{b}_gain"] = getattr(args, f"band{b}_gain")
            config[f"band{b}_q"] = getattr(args, f"band{b}_q")
            config[f"band{b}_type"] = getattr(args, f"band{b}_type")
            config[f"band{b}_skew"] = getattr(args, f"band{b}_skew")
            config[f"band{b}_dynamic"] = getattr(args, f"band{b}_dynamic")

        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            **config
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=========================================")
            print(f"  FAURGE PARAMETRIC EQ — PROCESSING REPORT")
            print(f"=========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Input RMS:        {result.get('input_rms_db', 0):.1f} dB")
            print(f"  Output RMS:       {result.get('output_rms_db', 0):.1f} dB")
            print(f"  Frames processed: {result.get('frames_processed', 0)}")
            print(f"=========================================\n")

    elif args.command == "quality":
        orig, sr1 = read_wav(args.original)
        proc, sr2 = read_wav(args.processed)
        report = quality_report(orig, proc, sr1)
        print(json.dumps(report, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
