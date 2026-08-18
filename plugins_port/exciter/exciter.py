"""
Portable Exciter — Faurge Portable Plugin
==========================================

Single-file self-contained dual-band harmonic exciter for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.

Usage
-----
    import exciter as ex

    # File-based API
    result = ex.process_file("input.wav", "output.wav",
                              high_drive_db=6.0, high_mix=0.5)

    # In-memory API
    audio_out, meta = ex.process(audio, sr,
                                 high_drive_db=6.0, high_mix=0.5,
                                 low_drive_db=3.0, low_mix=0.3)

    # Quality assessment
    report = ex.quality_report(original, processed, sr)
"""

import json
import time
import typing as T

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

PI = np.float64(np.pi)
SQRT2 = np.float64(1.4142135623730951)


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


# ── Biquad Utilities ────────────────────────────────────────────────────────


def _biquad_lp_coeffs(cutoff_hz: float, sample_rate: int) -> T.Tuple[
        np.ndarray, np.ndarray]:
    """Design 2nd-order Butterworth LPF biquad (b, a)."""
    w0 = 2.0 * PI * cutoff_hz / np.float64(sample_rate)
    alpha = np.sin(w0) / SQRT2
    b0 = (1.0 - np.cos(w0)) / 2.0
    b = np.array([b0, 1.0 - np.cos(w0), b0]) / (1.0 + alpha)
    a = np.array([1.0, -2.0 * np.cos(w0) / (1.0 + alpha),
                  (1.0 - alpha) / (1.0 + alpha)])
    return b, a


def _biquad_hp_coeffs(cutoff_hz: float, sample_rate: int) -> T.Tuple[
        np.ndarray, np.ndarray]:
    """Design 2nd-order Butterworth HPF biquad (b, a)."""
    w0 = 2.0 * PI * cutoff_hz / np.float64(sample_rate)
    alpha = np.sin(w0) / SQRT2
    b0 = (1.0 + np.cos(w0)) / 2.0
    b = np.array([b0, -(1.0 + np.cos(w0)), b0]) / (1.0 + alpha)
    a = np.array([1.0, -2.0 * np.cos(w0) / (1.0 + alpha),
                  (1.0 - alpha) / (1.0 + alpha)])
    return b, a


def _biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply a single biquad with zero initial state."""
    y, _ = lfilter(b, a, x, zi=np.zeros(2))
    return y


# ── Crossover Filter ────────────────────────────────────────────────────────


def _crossover_lr4(audio: np.ndarray, sample_rate: int,
                   crossover_hz: float) -> T.Tuple[np.ndarray, np.ndarray]:
    """Linkwitz-Riley 4th-order crossover (2x cascaded Butterworth).

    Matches crossover_filter.cpp. Returns (low_out, high_out).
    audio can be 1D (N,) or 2D (C, N).
    """
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]

    nyquist = sample_rate / 2.0
    n_samples = audio.shape[1]

    if crossover_hz <= 20.0:
        low = audio.copy()
        high = np.zeros_like(audio)
        return (low[0], high[0]) if was_1d else (low, high)

    if crossover_hz >= nyquist:
        low = np.zeros_like(audio)
        high = audio.copy()
        return (low[0], high[0]) if was_1d else (low, high)

    b_lp, a_lp = _biquad_lp_coeffs(crossover_hz, sample_rate)
    b_hp, a_hp = _biquad_hp_coeffs(crossover_hz, sample_rate)

    low = np.empty_like(audio)
    high = np.empty_like(audio)

    for ch in range(audio.shape[0]):
        x = audio[ch]
        tmp = _biquad(x, b_lp, a_lp)
        low[ch] = _biquad(tmp, b_lp, a_lp)
        tmp = _biquad(x, b_hp, a_hp)
        high[ch] = _biquad(tmp, b_hp, a_hp)

    if was_1d:
        return low[0], high[0]
    return low, high


# ── High Band Processor ─────────────────────────────────────────────────────


def _process_high_band(audio: np.ndarray, sample_rate: int,
                       drive_db: float) -> np.ndarray:
    """High-band harmonic saturator with 2x oversampling and AA filter.

    Matches high_band.cpp.
    audio can be 1D (N,) or 2D (C, N).
    """
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    n_channels, n_samples = audio.shape

    drive_linear = 10.0 ** (drive_db / 20.0)

    if drive_linear <= 0.0 or drive_db < 0.01:
        result = audio.copy()
        return result[0] if was_1d else result

    output = np.empty_like(audio)
    sos = butter(2, 0.45, output='sos')
    zi = np.zeros((sos.shape[0], 2))

    for ch in range(n_channels):
        x = audio[ch]

        # 2x ZOH oversample via broadcasting
        os_buf = np.repeat(x, 2)

        # Apply drive + tanh
        os_buf = np.tanh(os_buf * drive_linear)

        # Anti-alias filter: Butterworth 2nd-order LPF at fc=0.45
        os_filtered, _ = sosfilt(sos, os_buf, zi=zi.copy())

        # 2x downsample
        output[ch] = os_filtered[::2]

    return output[0] if was_1d else output


# ── Low Band Processor ──────────────────────────────────────────────────────


def _process_low_band(audio: np.ndarray, sample_rate: int,
                      drive_db: float, sub_level: float) -> np.ndarray:
    """Low-band sub-octave synthesizer via full-wave rectification + 4th-order LPF.

    Matches low_band.cpp.
    audio can be 1D (N,) or 2D (C, N).
    """
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    n_channels = audio.shape[0]

    drive_linear = 10.0 ** (drive_db / 20.0)

    if sub_level <= 0.0 and drive_db < 0.01:
        result = audio.copy()
        return result[0] if was_1d else result

    b, a = _biquad_lp_coeffs(120.0, sample_rate)
    output = np.empty_like(audio)

    for ch in range(n_channels):
        x = audio[ch]
        rect = np.abs(x) * drive_linear
        tmp = _biquad(rect, b, a)
        output[ch] = _biquad(tmp, b, a) * sub_level

    return output[0] if was_1d else output


# ── Metrics ─────────────────────────────────────────────────────────────────


def _peak_db(audio: np.ndarray) -> float:
    """Compute peak level in dBFS."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-30:
        return -120.0
    return float(20.0 * np.log10(peak))


def _rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dBFS."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-30:
        return -120.0
    return float(20.0 * np.log10(rms))


def _band_energy_db(audio: np.ndarray) -> float:
    """Compute band energy in dB (10*log10 of mean power)."""
    energy = float(np.mean(audio.astype(np.float64) ** 2))
    if energy < 1e-30:
        return -120.0
    return float(10.0 * np.log10(energy))


# ── Public API ──────────────────────────────────────────────────────────────


def process(
    audio: np.ndarray,
    sample_rate: int,
    high_drive_db: float = 3.0,
    high_mix: float = 0.50,
    high_crossover_hz: float = 2000.0,
    high_enable: bool = True,
    low_drive_db: float = 0.0,
    low_mix: float = 0.35,
    low_crossover_hz: float = 200.0,
    low_sub_level: float = 0.50,
    low_enable: bool = True,
    master_volume: float = 1.0,
) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the dual-band harmonic exciter.

    Args:
        audio: Input float32 array, shape (N,) for mono or (C, N) for multi-channel.
        sample_rate: Sample rate in Hz.
        high_drive_db: High-band pre-saturation gain in dB.
        high_mix: High-band wet/dry mix [0, 1].
        high_crossover_hz: High-band crossover frequency in Hz.
        high_enable: Enable high-band processing.
        low_drive_db: Low-band pre-rectification gain in dB.
        low_mix: Low-band wet/dry mix [0, 1].
        low_crossover_hz: Low-band crossover frequency in Hz.
        low_sub_level: Sub-octave injection level [0, 1].
        low_enable: Enable low-band processing.
        master_volume: Master volume multiplier.

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
            "high_band_energy_db": -120.0,
            "low_band_energy_db": -120.0,
            "frames_processed": 0,
        }

    # Process per-channel: audio is (N,) mono or (N, C) multichannel
    if was_1d:
        channels = [audio]
    else:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]

    outputs = []
    aggregate = None

    for idx, ch_audio in enumerate(channels):
        x = ch_audio[np.newaxis, :]  # (1, N)

        if idx == 0:
            input_peak_db = _peak_db(x)
            input_rms_db = _rms_db(x)

            _, high_buf_all = _crossover_lr4(x, sample_rate, high_crossover_hz)
            low_buf_all, _ = _crossover_lr4(x, sample_rate, low_crossover_hz)

            high_band_energy_db = _band_energy_db(high_buf_all)
            low_band_energy_db = _band_energy_db(low_buf_all)

        _, high_buf = _crossover_lr4(x, sample_rate, high_crossover_hz)
        low_buf, _ = _crossover_lr4(x, sample_rate, low_crossover_hz)

        if high_enable:
            high_out = _process_high_band(high_buf, sample_rate, high_drive_db)
        else:
            high_out = high_buf.copy()

        if low_enable:
            low_out = _process_low_band(low_buf, sample_rate,
                                         low_drive_db, low_sub_level)
        else:
            low_out = low_buf.copy()

        high_delta = high_out - high_buf
        low_delta = low_out - low_buf

        out = x + high_mix * high_delta + low_mix * low_delta
        out *= master_volume
        out = np.clip(out, -1.0, 1.0)
        outputs.append(out[0])

    output_peak_db = _peak_db(np.asarray(outputs))
    output_rms_db = _rms_db(np.asarray(outputs))

    if was_1d:
        audio = outputs[0]
    else:
        audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak_db,
        "output_peak_db": output_peak_db,
        "input_rms_db": input_rms_db,
        "output_rms_db": output_rms_db,
        "high_band_energy_db": high_band_energy_db,
        "low_band_energy_db": low_band_energy_db,
        "frames_processed": n_samples,
    }


def process_file(input_path: str, output_path: str,
                 high_drive_db: float = 3.0,
                 high_mix: float = 0.50,
                 high_crossover_hz: float = 2000.0,
                 high_enable: bool = True,
                 low_drive_db: float = 0.0,
                 low_mix: float = 0.35,
                 low_crossover_hz: float = 200.0,
                 low_sub_level: float = 0.50,
                 low_enable: bool = True,
                 master_volume: float = 1.0,
                 verbose: bool = False) -> dict:
    """Read WAV → process → write WAV.

    Processes each channel independently, then aggregates.
    """
    audio, sr = read_wav(input_path)

    if verbose:
        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        print(f"[exciter] Input: {input_path}", file=__import__('sys').stderr)
        print(f"[exciter]   Channels:    {n_channels}", file=__import__('sys').stderr)
        print(f"[exciter]   Sample rate: {sr} Hz", file=__import__('sys').stderr)
        print(f"[exciter]   Frames:      {audio.shape[-1]}", file=__import__('sys').stderr)

    if audio.ndim == 1:
        processed, result = process(
            audio, sr,
            high_drive_db=high_drive_db,
            high_mix=high_mix,
            high_crossover_hz=high_crossover_hz,
            high_enable=high_enable,
            low_drive_db=low_drive_db,
            low_mix=low_mix,
            low_crossover_hz=low_crossover_hz,
            low_sub_level=low_sub_level,
            low_enable=low_enable,
            master_volume=master_volume,
        )
        write_wav(output_path, processed, sr)
        return result

    # Multi-channel: process each channel
    channels = [audio[:, ch] for ch in range(audio.shape[1])]
    aggregate = {}
    for idx, ch_audio in enumerate(channels):
        if verbose:
            print(f"[exciter] Processing channel {idx + 1}/{len(channels)}...",
                  file=__import__('sys').stderr)
        _, ch_result = process(
            ch_audio, sr,
            high_drive_db=high_drive_db,
            high_mix=high_mix,
            high_crossover_hz=high_crossover_hz,
            high_enable=high_enable,
            low_drive_db=low_drive_db,
            low_mix=low_mix,
            low_crossover_hz=low_crossover_hz,
            low_sub_level=low_sub_level,
            low_enable=low_enable,
            master_volume=master_volume,
        )
        if idx == 0:
            aggregate = ch_result
        else:
            aggregate["high_band_energy_db"] = max(
                aggregate["high_band_energy_db"],
                ch_result["high_band_energy_db"])
            aggregate["low_band_energy_db"] = max(
                aggregate["low_band_energy_db"],
                ch_result["low_band_energy_db"])
            aggregate["frames_processed"] += ch_result["frames_processed"]

    processed_audio = np.column_stack([
        process(ch_audio, sr)[0] for ch_audio in channels
    ])
    write_wav(output_path, processed_audio, sr)

    if verbose:
        print(f"[exciter] Output written: {output_path}",
              file=__import__('sys').stderr)

    return aggregate


def quality_report(original: np.ndarray, processed: np.ndarray,
                   sample_rate: int) -> dict:
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

    snr_db = (float(10.0 * np.log10(sig_power / noise_power))
              if noise_power > 1e-30 else float('inf'))

    before_power = np.mean(orig ** 2)
    after_power = np.mean(proc ** 2)
    improvement_db = (float(10.0 * np.log10(after_power / before_power))
                      if before_power > 1e-30 else 0.0)

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
    parser = argparse.ArgumentParser(
        description="Portable Exciter — Dual-Band Harmonic Exciter")
    sub = parser.add_subparsers(dest="command")

    excite_cmd = sub.add_parser("excite", help="Process audio with exciter")
    excite_cmd.add_argument("input", type=str)
    excite_cmd.add_argument("output", type=str)
    excite_cmd.add_argument("--high-drive", type=float, default=3.0)
    excite_cmd.add_argument("--high-mix", type=float, default=0.50)
    excite_cmd.add_argument("--high-cross", type=float, default=2000.0)
    excite_cmd.add_argument("--low-drive", type=float, default=0.0)
    excite_cmd.add_argument("--low-mix", type=float, default=0.35)
    excite_cmd.add_argument("--low-cross", type=float, default=200.0)
    excite_cmd.add_argument("--low-sub", type=float, default=0.50)
    excite_cmd.add_argument("--no-high", action="store_false", dest="high_enable")
    excite_cmd.add_argument("--no-low", action="store_false", dest="low_enable")
    excite_cmd.add_argument("--volume", type=float, default=1.0)
    excite_cmd.add_argument("--json", action="store_true")
    excite_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "excite":
        result = process_file(
            args.input, args.output,
            high_drive_db=args.high_drive,
            high_mix=args.high_mix,
            high_crossover_hz=args.high_cross,
            high_enable=args.high_enable,
            low_drive_db=args.low_drive,
            low_mix=args.low_mix,
            low_crossover_hz=args.low_cross,
            low_sub_level=args.low_sub,
            low_enable=args.low_enable,
            master_volume=args.volume,
            verbose=args.verbose,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Success: {result.get('success')}")
            print(f"Time: {result.get('processing_time_ms', 0):.1f} ms")
            print(f"Input peak: {result.get('input_peak_db', 0):.1f} dB")
            print(f"Output peak: {result.get('output_peak_db', 0):.1f} dB")
            print(f"High band energy: {result.get('high_band_energy_db', 0):.1f} dB")
            print(f"Low band energy: {result.get('low_band_energy_db', 0):.1f} dB")

    elif args.command == "quality":
        orig, sr1 = read_wav(args.original)
        proc, sr2 = read_wav(args.processed)
        report = quality_report(orig, proc, sr1)
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
