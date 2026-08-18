"""
Portable Limiter — Faurge Portable Plugin
==========================================

Single-file self-contained peak limiter for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production limiter exactly.

Usage
-----
    import limiter

    # In-memory API
    audio_out, meta = limiter.process(audio, sr, ceiling_db=-1.0, release_ms=100.0)
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
    "ceiling_db":       -1.0,
    "release_ms":       100.0,
    "lookahead_ms":     5.0,
    "clip_mode":        "soft",
    "stereo_link":      1.0,
    "oversampling":     1,
}

CLIP_MODE_MAP = {
    0: "hard",
    "hard": "hard",
    1: "soft",
    "soft": "soft",
}


# ── Peak Predictor (circular buffer lookahead) ──────────────────────────────


class PeakPredictor:
    def __init__(self, lookahead_samples: int):
        self.size = max(lookahead_samples, 1)
        self.buf = [0.0] * self.size
        self.write_idx = 0

    def process_sample(self, x: float) -> T.Tuple[float, float]:
        delayed = self.buf[(self.write_idx + 1) % self.size]
        self.buf[self.write_idx] = x
        predicted = max(abs(v) for v in self.buf)
        self.write_idx = (self.write_idx + 1) % self.size
        return delayed, predicted


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


def _linear_to_db(linear: float) -> float:
    if linear < 1e-30:
        return -120.0
    return 20.0 * np.log10(linear)


# ── Processing Core ──────────────────────────────────────────────────────────


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _limiter_mono_jit(out, n, ceiling_lin, release_alpha, clip_mode_is_soft,
                          tanh_scale, buf, write_idx, size):
        smoothed_gr_db = 0.0
        max_gr = 0.0
        sum_gr = 0.0
        clipped_samples = 0

        # Initialize running max
        current_max = 0.0
        max_pos = 0
        for k in range(size):
            if abs(buf[k]) > current_max:
                current_max = abs(buf[k])
                max_pos = k

        for i in range(n):
            # Write new sample
            buf[write_idx] = out[i]

            # Update running max with new sample
            abs_new = abs(out[i])
            if abs_new >= current_max:
                current_max = abs_new
                max_pos = write_idx

            # Read delayed sample
            read_pos = (write_idx + 1) % size
            delayed = buf[read_pos]

            # If delayed sample was the max, recompute (it will be overwritten next write)
            if read_pos == max_pos:
                current_max = 0.0
                max_pos = 0
                for k in range(size):
                    if abs(buf[k]) > current_max:
                        current_max = abs(buf[k])
                        max_pos = k

            predicted_peak = current_max
            write_idx = (write_idx + 1) % size

            # Limiter logic
            if predicted_peak <= ceiling_lin:
                target_gr_db = 0.0
            else:
                if ceiling_lin / predicted_peak < 1e-30:
                    target_gr_db = -120.0
                else:
                    target_gr_db = 20.0 * np.log10(ceiling_lin / predicted_peak)

            if target_gr_db < smoothed_gr_db:
                smoothed_gr_db = target_gr_db
            else:
                smoothed_gr_db += release_alpha * (target_gr_db - smoothed_gr_db)

            gain_lin = 10.0 ** (smoothed_gr_db / 20.0)
            out_sample = delayed * gain_lin

            if clip_mode_is_soft:
                scaled = out_sample / ceiling_lin
                out_sample = ceiling_lin * tanh_scale * np.tanh(scaled / tanh_scale)
            else:
                out_sample = max(-ceiling_lin, min(ceiling_lin, out_sample))
                if abs(out_sample) >= ceiling_lin:
                    clipped_samples += 1

            out[i] = out_sample

            abs_gr = abs(smoothed_gr_db)
            sum_gr += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

        return max_gr, sum_gr, clipped_samples, write_idx

    @numba.njit(cache=True)
    def _limiter_stereo_jit(out, n, ceiling_lin, release_alpha, clip_mode_is_soft,
                            tanh_scale, link,
                            buf0, write_idx0, buf1, write_idx1, size):
        smoothed_gr_db = 0.0
        max_gr = 0.0
        sum_gr = 0.0
        clipped_samples = 0

        # Init running max for both channels
        current_max0 = 0.0
        max_pos0 = 0
        for k in range(size):
            if abs(buf0[k]) > current_max0:
                current_max0 = abs(buf0[k])
                max_pos0 = k

        current_max1 = 0.0
        max_pos1 = 0
        for k in range(size):
            if abs(buf1[k]) > current_max1:
                current_max1 = abs(buf1[k])
                max_pos1 = k

        for i in range(n):
            # Channel 0
            buf0[write_idx0] = out[i, 0]
            abs_new = abs(out[i, 0])
            if abs_new >= current_max0:
                current_max0 = abs_new
                max_pos0 = write_idx0
            read_pos0 = (write_idx0 + 1) % size
            delayed0 = buf0[read_pos0]
            if read_pos0 == max_pos0:
                current_max0 = 0.0
                max_pos0 = 0
                for k in range(size):
                    if abs(buf0[k]) > current_max0:
                        current_max0 = abs(buf0[k])
                        max_pos0 = k
            predicted0 = current_max0
            write_idx0 = (write_idx0 + 1) % size

            # Channel 1
            buf1[write_idx1] = out[i, 1]
            abs_new = abs(out[i, 1])
            if abs_new >= current_max1:
                current_max1 = abs_new
                max_pos1 = write_idx1
            read_pos1 = (write_idx1 + 1) % size
            delayed1 = buf1[read_pos1]
            if read_pos1 == max_pos1:
                current_max1 = 0.0
                max_pos1 = 0
                for k in range(size):
                    if abs(buf1[k]) > current_max1:
                        current_max1 = abs(buf1[k])
                        max_pos1 = k
            predicted1 = current_max1
            write_idx1 = (write_idx1 + 1) % size

            combined_peak = predicted0 * link + predicted1 * (1.0 - link)

            if combined_peak <= ceiling_lin:
                target_gr_db = 0.0
            else:
                if ceiling_lin / combined_peak < 1e-30:
                    target_gr_db = -120.0
                else:
                    target_gr_db = 20.0 * np.log10(ceiling_lin / combined_peak)

            if target_gr_db < smoothed_gr_db:
                smoothed_gr_db = target_gr_db
            else:
                smoothed_gr_db += release_alpha * (target_gr_db - smoothed_gr_db)

            gain_lin = 10.0 ** (smoothed_gr_db / 20.0)

            for ch in range(2):
                if ch == 0:
                    ds = delayed0
                else:
                    ds = delayed1
                out_sample = ds * gain_lin
                if clip_mode_is_soft:
                    scaled = out_sample / ceiling_lin
                    out_sample = ceiling_lin * tanh_scale * np.tanh(scaled / tanh_scale)
                else:
                    out_sample = max(-ceiling_lin, min(ceiling_lin, out_sample))
                    if abs(out_sample) >= ceiling_lin:
                        clipped_samples += 1
                out[i, ch] = out_sample

            abs_gr = abs(smoothed_gr_db)
            sum_gr += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

        return max_gr, sum_gr, clipped_samples, write_idx0, write_idx1


def _process_mono(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process a mono channel through the limiter."""
    n = len(audio)
    out = np.array(audio, dtype=np.float64)

    ceiling_lin = _db_to_linear(cfg["ceiling_db"])
    lookahead_samples = int(cfg["lookahead_ms"] * float(sr) * 0.001)
    release_alpha = 1.0 - np.exp(-2.2 / max(cfg["release_ms"] * float(sr) * 0.001, 1.0))
    release_alpha = min(1.0, max(0.0, release_alpha))
    tanh_scale = 1.313
    clip_mode = CLIP_MODE_MAP.get(cfg["clip_mode"], "soft")

    if NUMBA_AVAILABLE:
        out = np.array(audio, dtype=np.float64)
        size = max(lookahead_samples, 1)
        buf = np.zeros(size, dtype=np.float64)
        max_gr, sum_gr, clipped_samples, _ = _limiter_mono_jit(
            out, n, ceiling_lin, release_alpha, clip_mode == "soft",
            tanh_scale, buf, 0, size,
        )
    else:
        out = np.array(audio, dtype=np.float64)
        predictor = PeakPredictor(lookahead_samples)
        smoothed_gr_db = 0.0
        max_gr = 0.0
        sum_gr = 0.0
        clipped_samples = 0

        for i in range(n):
            delayed, predicted_peak = predictor.process_sample(out[i])

            if predicted_peak <= ceiling_lin:
                target_gr_db = 0.0
            else:
                target_gr_db = _linear_to_db(ceiling_lin / predicted_peak)

            if target_gr_db < smoothed_gr_db:
                smoothed_gr_db = target_gr_db
            else:
                smoothed_gr_db += release_alpha * (target_gr_db - smoothed_gr_db)

            gain_lin = _db_to_linear(smoothed_gr_db)
            out_sample = delayed * gain_lin

            if clip_mode == "soft":
                scaled = out_sample / ceiling_lin
                out_sample = ceiling_lin * tanh_scale * np.tanh(scaled / tanh_scale)
            else:
                out_sample = np.clip(out_sample, -ceiling_lin, ceiling_lin)
                if abs(out_sample) >= ceiling_lin:
                    clipped_samples += 1

            out[i] = out_sample

            abs_gr = abs(smoothed_gr_db)
            sum_gr += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

    return out, {
        "max_gain_reduction_db": max_gr,
        "avg_gain_reduction_db": sum_gr / max(n, 1),
        "clipped_samples": clipped_samples,
    }


def _process_stereo(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process stereo with stereo linking."""
    n, n_ch = audio.shape
    out = np.array(audio, dtype=np.float64)

    ceiling_lin = _db_to_linear(cfg["ceiling_db"])
    lookahead_samples = int(cfg["lookahead_ms"] * float(sr) * 0.001)
    release_alpha = 1.0 - np.exp(-2.2 / max(cfg["release_ms"] * float(sr) * 0.001, 1.0))
    release_alpha = min(1.0, max(0.0, release_alpha))
    tanh_scale = 1.313
    clip_mode = CLIP_MODE_MAP.get(cfg["clip_mode"], "soft")
    link = float(cfg["stereo_link"])

    if NUMBA_AVAILABLE:
        out = np.array(audio, dtype=np.float64)
        size = max(lookahead_samples, 1)
        buf0 = np.zeros(size, dtype=np.float64)
        buf1 = np.zeros(size, dtype=np.float64)
        max_gr, sum_gr, clipped_samples, _, _ = _limiter_stereo_jit(
            out, n, ceiling_lin, release_alpha, clip_mode == "soft",
            tanh_scale, link, buf0, 0, buf1, 0, size,
        )
    else:
        out = np.array(audio, dtype=np.float64)
        predictors = [PeakPredictor(lookahead_samples), PeakPredictor(lookahead_samples)]
        smoothed_gr_db = 0.0
        max_gr = 0.0
        sum_gr = 0.0
        clipped_samples = 0

        for i in range(n):
            delayed = [0.0, 0.0]
            predicted = [0.0, 0.0]
            for ch in range(2):
                delayed[ch], predicted[ch] = predictors[ch].process_sample(out[i, ch])

            combined_peak = predicted[0] * link + predicted[1] * (1.0 - link)

            if combined_peak <= ceiling_lin:
                target_gr_db = 0.0
            else:
                target_gr_db = _linear_to_db(ceiling_lin / combined_peak)

            if target_gr_db < smoothed_gr_db:
                smoothed_gr_db = target_gr_db
            else:
                smoothed_gr_db += release_alpha * (target_gr_db - smoothed_gr_db)

            gain_lin = _db_to_linear(smoothed_gr_db)

            for ch in range(2):
                out_sample = delayed[ch] * gain_lin
                if clip_mode == "soft":
                    scaled = out_sample / ceiling_lin
                    out_sample = ceiling_lin * tanh_scale * np.tanh(scaled / tanh_scale)
                else:
                    out_sample = np.clip(out_sample, -ceiling_lin, ceiling_lin)
                    if abs(out_sample) >= ceiling_lin:
                        clipped_samples += 1
                out[i, ch] = out_sample

            abs_gr = abs(smoothed_gr_db)
            sum_gr += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

    return out, {
        "max_gain_reduction_db": max_gr,
        "avg_gain_reduction_db": sum_gr / max(n, 1),
        "clipped_samples": clipped_samples,
    }


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the peak limiter.

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
            if k_clean == "clip_mode":
                cfg[k_clean] = CLIP_MODE_MAP.get(v, "soft")
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
            "max_gain_reduction_db": 0.0,
            "avg_gain_reduction_db": 0.0,
            "clipped_samples": 0,
            "frames_processed": 0,
        }

    input_peak = _peak_db(audio)
    input_rms = _rms_db(audio)

    is_stereo = not was_1d and audio.shape[1] == 2
    if is_stereo and cfg["stereo_link"] > 0.0:
        processed, lim_info = _process_stereo(audio, sr, cfg)
    elif was_1d:
        processed, lim_info = _process_mono(audio, sr, cfg)
    else:
        n_ch = audio.shape[1]
        outputs = []
        lim_info = {"max_gain_reduction_db": 0.0, "avg_gain_reduction_db": 0.0, "clipped_samples": 0}
        for ch in range(n_ch):
            ch_out, ch_info = _process_mono(audio[:, ch], sr, cfg)
            outputs.append(ch_out)
            if ch_info["max_gain_reduction_db"] > lim_info["max_gain_reduction_db"]:
                lim_info["max_gain_reduction_db"] = ch_info["max_gain_reduction_db"]
            lim_info["clipped_samples"] += ch_info["clipped_samples"]
        processed = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": _peak_db(processed),
        "input_rms_db": input_rms,
        "output_rms_db": _rms_db(processed),
        "max_gain_reduction_db": lim_info["max_gain_reduction_db"],
        "avg_gain_reduction_db": lim_info["avg_gain_reduction_db"],
        "clipped_samples": lim_info["clipped_samples"],
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
        print(f"[limiter] Input: {input_path}", file=sys.stderr)
        print(f"[limiter]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[limiter]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[limiter]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[limiter] Output written: {output_path}", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Portable Limiter — Peak Limiter")
    sub = parser.add_subparsers(dest="command")

    lim_cmd = sub.add_parser("limit", help="Limit audio")
    lim_cmd.add_argument("input", type=str)
    lim_cmd.add_argument("output", type=str)
    lim_cmd.add_argument("--ceiling", type=float, default=-1.0)
    lim_cmd.add_argument("--release", type=float, default=100.0)
    lim_cmd.add_argument("--lookahead", type=float, default=5.0)
    lim_cmd.add_argument("--clip-mode", type=str, default="soft", choices=["soft", "hard"])
    lim_cmd.add_argument("--stereo-link", type=float, default=1.0)
    lim_cmd.add_argument("--oversampling", type=int, default=1, choices=[1, 2, 4])
    lim_cmd.add_argument("--json", action="store_true")
    lim_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "limit":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            ceiling_db=args.ceiling,
            release_ms=args.release,
            lookahead_ms=args.lookahead,
            clip_mode=args.clip_mode,
            stereo_link=args.stereo_link,
            oversampling=args.oversampling,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE LIMITER — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Max GR:           {result.get('max_gain_reduction_db', 0):.1f} dB")
            print(f"  Clipped samples:  {result.get('clipped_samples', 0)}")
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
