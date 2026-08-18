"""
Portable Compressor — Faurge Portable Plugin
=============================================

Single-file self-contained dynamic range compressor for Kaggle and local use.
Pure NumPy/SciPy — no C++ build step, no model download.
Matches the behavior and mechanics of the C++ production compressor exactly.

Usage
-----
    import compressor

    # File-based API
    result = compressor.process_file("input.wav", "output.wav",
                                     threshold_db=-24.0, ratio=4.0)

    # In-memory API
    audio_out, meta = compressor.process(audio, sr, threshold_db=-24.0, ratio=4.0)
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
    "threshold_db":      -24.0,
    "ratio":             4.0,
    "attack_ms":         5.0,
    "release_ms":        150.0,
    "knee_db":           6.0,
    "lookahead_ms":      0.0,
    "hold_ms":           0.0,
    "wet_dry_mix":       1.0,
    "stereo_link":       0.0,   # Default 0 for mono training
    "sidechain_hp_hz":   20.0,
    "sidechain_lp_hz":   20000.0,
    "saturate_drive_db": 0.0,
    "output_trim_db":    0.0,
    "detector_type":     "RMS",  # RMS, peak, feed_forward, feed_back
}

DETECTOR_TYPE_MAP = {
    0: "RMS",
    "RMS": "RMS",
    "rms": "RMS",
    1: "peak",
    "peak": "peak",
    2: "feed_forward",
    "feed_forward": "feed_forward",
    3: "feed_back",
    "feed_back": "feed_back",
}


# ── Biquad Helpers (sidechain HP/LP) ────────────────────────────────────────


def _design_hp(freq: float, sr: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """Design highpass biquad (Q=0.7071, standard RBJ)."""
    w0 = 2.0 * np.pi * freq / float(sr)
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0
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


def _design_lp(freq: float, sr: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """Design lowpass biquad (Q=0.7071, standard RBJ)."""
    w0 = 2.0 * np.pi * freq / float(sr)
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0
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


def _biquad_sample(b: np.ndarray, a: np.ndarray, state: list, x: float) -> float:
    """Process one sample through a biquad (Direct Form I)."""
    y = b[0] * x + b[1] * state[0] + b[2] * state[1] \
        - a[1] * state[2] - a[2] * state[3]
    state[1] = state[0]
    state[0] = x
    state[3] = state[2]
    state[2] = y
    return y


# ── Envelope Follower ────────────────────────────────────────────────────────


def _rc_alpha(tau_ms: float, sr: int) -> float:
    """Compute RC time constant alpha, matching C++ 2.2× factor."""
    if tau_ms <= 0.0 or sr <= 0:
        return 1.0
    samples = tau_ms * float(sr) * 0.001
    return 1.0 - np.exp(-2.2 / samples)


# ── Gain Computer ────────────────────────────────────────────────────────────


def _compute_gain_db(env_linear: float, threshold_db: float,
                     ratio: float, knee_db: float) -> float:
    """Static gain curve with soft knee, matching C++ exactly."""
    eps = 1e-30
    env_db = 20.0 * np.log10(max(env_linear, eps))
    overshoot = env_db - threshold_db

    if overshoot <= -knee_db * 0.5:
        return 0.0

    if overshoot >= knee_db * 0.5:
        gr = -overshoot * (ratio - 1.0) / ratio
    else:
        x = overshoot / knee_db + 0.5
        x = max(0.0, min(1.0, x))
        gr = -x * x * knee_db * 0.5 * (ratio - 1.0) / ratio

    return max(-120.0, min(0.0, gr))


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
    def _biquad_jit(b0, b1, b2, a1, a2, state, x):
        y = b0 * x + b1 * state[0] + b2 * state[1] - a1 * state[2] - a2 * state[3]
        state[1] = state[0]
        state[0] = x
        state[3] = state[2]
        state[2] = y
        return y

    @numba.njit(cache=True)
    def _rc_alpha_jit(tau_ms, sr):
        if tau_ms <= 0.0 or sr <= 0:
            return 1.0
        samples = tau_ms * float(sr) * 0.001
        return 1.0 - np.exp(-2.2 / samples)

    @numba.njit(cache=True)
    def _compute_gain_db_jit(env_linear, threshold_db, ratio, knee_db):
        eps = 1e-30
        env_db = 20.0 * np.log10(max(env_linear, eps))
        overshoot = env_db - threshold_db

        if overshoot <= -knee_db * 0.5:
            return 0.0

        if overshoot >= knee_db * 0.5:
            gr = -overshoot * (ratio - 1.0) / ratio
        else:
            x = overshoot / knee_db + 0.5
            x = max(0.0, min(1.0, x))
            gr = -x * x * knee_db * 0.5 * (ratio - 1.0) / ratio

        return max(-120.0, min(0.0, gr))

    @numba.njit(cache=True)
    def _process_mono_jit(out, n, sr,
                          sc_hp_b0, sc_hp_b1, sc_hp_b2, sc_hp_a1, sc_hp_a2,
                          sc_lp_b0, sc_lp_b1, sc_lp_b2, sc_lp_a1, sc_lp_a2,
                          attack_ms, release_ms, threshold_db, ratio, knee_db,
                          lookahead_samples, hold_ms, wet_dry_mix,
                          saturate_drive_lin, output_trim_lin, detector_is_rms):
        sc_hp_state = np.zeros(4, dtype=np.float64)
        sc_lp_state = np.zeros(4, dtype=np.float64)

        la_size = max(lookahead_samples, 1)
        env_delay_buf = np.zeros(la_size, dtype=np.float64)
        audio_delay_buf = np.zeros(la_size, dtype=np.float64)
        delay_write_idx = 0

        envelope = 0.0
        smoothed_gain_db = 0.0
        hold_timer = 0
        hold_gain_db = 0.0

        if hold_ms > 0.0 and sr > 0:
            hold_samples = int(hold_ms * float(sr) * 0.001)
        else:
            hold_samples = 0

        attack_alpha = _rc_alpha_jit(attack_ms, sr)
        release_alpha = _rc_alpha_jit(release_ms, sr)

        sum_gr_abs = 0.0
        max_gr = 0.0

        for i in range(n):
            # Sidechain filtering (inlined biquad x2)
            sc_sample = _biquad_jit(sc_hp_b0, sc_hp_b1, sc_hp_b2, sc_hp_a1, sc_hp_a2, sc_hp_state, out[i])
            sc_sample = _biquad_jit(sc_lp_b0, sc_lp_b1, sc_lp_b2, sc_lp_a1, sc_lp_a2, sc_lp_state, sc_sample)

            # Envelope detection
            if detector_is_rms:
                detected = sc_sample * sc_sample
            else:
                detected = abs(sc_sample)

            if detected > envelope:
                alpha = attack_alpha
            else:
                alpha = release_alpha

            envelope += alpha * (detected - envelope)

            if detector_is_rms:
                env = np.sqrt(max(envelope, 0.0))
            else:
                env = envelope

            # Lookahead delay
            if lookahead_samples > 0:
                env_delay_buf[delay_write_idx] = env
                delayed_env = env_delay_buf[(delay_write_idx + 1) % la_size]
                env = delayed_env

                audio_delay_buf[delay_write_idx] = out[i]
                out[i] = audio_delay_buf[(delay_write_idx + 1) % la_size]

                delay_write_idx = (delay_write_idx + 1) % la_size

            # Gain computer — static curve
            gr_db = _compute_gain_db_jit(env, threshold_db, ratio, knee_db)

            # Gain computer — smoothing with hold
            if gr_db < smoothed_gain_db - 0.001:
                hold_timer = 0
                a = min(1.0, attack_alpha)
                smoothed_gain_db += a * (gr_db - smoothed_gain_db)
                hold_gain_db = smoothed_gain_db
            elif gr_db > smoothed_gain_db + 0.001:
                if hold_timer > 0:
                    hold_timer -= 1
                    smoothed_gain_db = hold_gain_db
                else:
                    a = min(1.0, release_alpha)
                    smoothed_gain_db += a * (gr_db - smoothed_gain_db)

            if hold_samples > 0 and hold_timer == 0 and abs(gr_db - smoothed_gain_db) < 0.001:
                hold_timer = hold_samples
                hold_gain_db = smoothed_gain_db

            # Apply gain
            gain_lin = 10.0 ** (smoothed_gain_db / 20.0)
            out_sample = out[i] * gain_lin

            # Saturation
            if saturate_drive_lin > 1.0:
                x = out_sample * saturate_drive_lin
                x = np.tanh(x)
                out_sample = x / saturate_drive_lin

            # Wet/dry mix
            out_sample = out_sample * wet_dry_mix + out[i] * (1.0 - wet_dry_mix)

            # Output trim
            out_sample *= output_trim_lin

            out[i] = out_sample

            abs_gr = abs(smoothed_gain_db)
            sum_gr_abs += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

        return max_gr, sum_gr_abs


def _process_mono(audio: np.ndarray, sr: int, cfg: dict) -> T.Tuple[np.ndarray, dict]:
    """Process a mono channel through the compressor, sample-by-sample."""
    n = len(audio)
    out = np.array(audio, dtype=np.float64)

    # Sidechain filters
    sc_hp_b, sc_hp_a = _design_hp(cfg["sidechain_hp_hz"], sr)
    sc_lp_b, sc_lp_a = _design_lp(cfg["sidechain_lp_hz"], sr)

    # Lookahead ring buffers
    lookahead_samples = int(cfg["lookahead_ms"] * float(sr) * 0.001)

    # Pre-compute
    output_trim_lin = _db_to_linear(cfg["output_trim_db"])
    saturate_drive_lin = _db_to_linear(cfg["saturate_drive_db"])
    detector_type = DETECTOR_TYPE_MAP.get(cfg["detector_type"], "RMS")

    if NUMBA_AVAILABLE:
        out = np.array(audio, dtype=np.float64)
        max_gr, sum_gr_abs = _process_mono_jit(
            out, n, sr,
            sc_hp_b[0], sc_hp_b[1], sc_hp_b[2], sc_hp_a[1], sc_hp_a[2],
            sc_lp_b[0], sc_lp_b[1], sc_lp_b[2], sc_lp_a[1], sc_lp_a[2],
            cfg["attack_ms"], cfg["release_ms"], cfg["threshold_db"], cfg["ratio"], cfg["knee_db"],
            lookahead_samples, cfg["hold_ms"], cfg["wet_dry_mix"],
            saturate_drive_lin, output_trim_lin, detector_type == "RMS",
        )
    else:
        out = np.array(audio, dtype=np.float64)
        sc_hp_state = [0.0, 0.0, 0.0, 0.0]
        sc_lp_state = [0.0, 0.0, 0.0, 0.0]
        env_delay_buf = [0.0] * max(lookahead_samples, 1)
        audio_delay_buf = [0.0] * max(lookahead_samples, 1)
        delay_write_idx = 0
        envelope = 0.0
        smoothed_gain_db = 0.0
        hold_timer = 0
        hold_gain_db = 0.0
        hold_samples = 0
        sum_gr_abs = 0.0
        max_gr = 0.0

        for i in range(n):
            sc_sample = _biquad_sample(sc_hp_b, sc_hp_a, sc_hp_state, out[i])
            sc_sample = _biquad_sample(sc_lp_b, sc_lp_a, sc_lp_state, sc_sample)

            if detector_type == "RMS":
                detected = sc_sample * sc_sample
            else:
                detected = abs(sc_sample)

            if detected > envelope:
                alpha = _rc_alpha(cfg["attack_ms"], sr)
            else:
                alpha = _rc_alpha(cfg["release_ms"], sr)

            envelope += alpha * (detected - envelope)

            if detector_type == "RMS":
                env = np.sqrt(max(envelope, 0.0))
            else:
                env = envelope

            if lookahead_samples > 0:
                env_delay_buf[delay_write_idx] = env
                delayed_env = env_delay_buf[(delay_write_idx + 1) % lookahead_samples]
                env = delayed_env
                audio_delay_buf[delay_write_idx] = out[i]
                out[i] = audio_delay_buf[(delay_write_idx + 1) % lookahead_samples]
                delay_write_idx = (delay_write_idx + 1) % lookahead_samples

            gr_db = _compute_gain_db(env, cfg["threshold_db"], cfg["ratio"], cfg["knee_db"])

            if cfg["hold_ms"] > 0.0 and sr > 0:
                hold_samples = int(cfg["hold_ms"] * float(sr) * 0.001)
            else:
                hold_samples = 0

            if gr_db < smoothed_gain_db - 0.001:
                hold_timer = 0
                a = _rc_alpha(cfg["attack_ms"], sr)
                a = min(1.0, a)
                smoothed_gain_db += a * (gr_db - smoothed_gain_db)
                hold_gain_db = smoothed_gain_db
            elif gr_db > smoothed_gain_db + 0.001:
                if hold_timer > 0:
                    hold_timer -= 1
                    smoothed_gain_db = hold_gain_db
                else:
                    a = _rc_alpha(cfg["release_ms"], sr)
                    a = min(1.0, a)
                    smoothed_gain_db += a * (gr_db - smoothed_gain_db)

            if hold_samples > 0 and hold_timer == 0 and abs(gr_db - smoothed_gain_db) < 0.001:
                hold_timer = hold_samples
                hold_gain_db = smoothed_gain_db

            gain_lin = _db_to_linear(smoothed_gain_db)
            out_sample = out[i] * gain_lin

            if cfg["saturate_drive_db"] > 0.0:
                x = out_sample * saturate_drive_lin
                x = np.tanh(x)
                out_sample = x / saturate_drive_lin

            out_sample = out_sample * cfg["wet_dry_mix"] + out[i] * (1.0 - cfg["wet_dry_mix"])
            out_sample *= output_trim_lin
            out[i] = out_sample

            abs_gr = abs(smoothed_gain_db)
            sum_gr_abs += abs_gr
            if abs_gr > max_gr:
                max_gr = abs_gr

    return out, {
        "gain_reduction_db": max_gr,
        "avg_gain_reduction_db": sum_gr_abs / max(n, 1),
    }


# ── Public API ───────────────────────────────────────────────────────────────


def process(audio: np.ndarray, sr: int, **config) -> T.Tuple[np.ndarray, dict]:
    """Process audio through the dynamic range compressor.

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
            if k_clean == "detector_type":
                cfg[k_clean] = DETECTOR_TYPE_MAP.get(v, "RMS")
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
            "gain_reduction_db": 0.0,
            "avg_gain_reduction_db": 0.0,
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
        gr_info = {"gain_reduction_db": 0.0, "avg_gain_reduction_db": 0.0}
        for ch in range(n_ch):
            ch_out, ch_gr = _process_mono(audio[:, ch], sr, cfg)
            outputs.append(ch_out)
            if ch_gr["gain_reduction_db"] > gr_info["gain_reduction_db"]:
                gr_info = ch_gr
        processed_audio = np.column_stack(outputs)

    elapsed_ms = (time.time() - t0) * 1000.0

    return processed_audio.astype(np.float32), {
        "success": True,
        "processing_time_ms": elapsed_ms,
        "input_peak_db": input_peak,
        "output_peak_db": _peak_db(processed_audio),
        "input_rms_db": input_rms,
        "output_rms_db": _rms_db(processed_audio),
        "gain_reduction_db": gr_info["gain_reduction_db"],
        "avg_gain_reduction_db": gr_info["avg_gain_reduction_db"],
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
        print(f"[compress] Input: {input_path}", file=sys.stderr)
        print(f"[compress]   Channels:    {n_channels}", file=sys.stderr)
        print(f"[compress]   Sample rate: {sr} Hz", file=sys.stderr)
        print(f"[compress]   Frames:      {audio.shape[0]}", file=sys.stderr)

    processed, result = process(audio, sr, **config)

    if output_path is not None:
        write_wav(output_path, processed, sr)
        if verbose:
            print(f"[compress] Output written: {output_path}", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="Portable Compressor — Dynamic Range Compressor")
    sub = parser.add_subparsers(dest="command")

    comp_cmd = sub.add_parser("compress", help="Compress audio")
    comp_cmd.add_argument("input", type=str)
    comp_cmd.add_argument("output", type=str)
    comp_cmd.add_argument("--threshold", type=float, default=-24.0)
    comp_cmd.add_argument("--ratio", type=float, default=4.0)
    comp_cmd.add_argument("--attack", type=float, default=5.0)
    comp_cmd.add_argument("--release", type=float, default=150.0)
    comp_cmd.add_argument("--knee", type=float, default=6.0)
    comp_cmd.add_argument("--lookahead", type=float, default=0.0)
    comp_cmd.add_argument("--hold", type=float, default=0.0)
    comp_cmd.add_argument("--wet-dry", type=float, default=1.0)
    comp_cmd.add_argument("--stereo-link", type=float, default=0.0)
    comp_cmd.add_argument("--sidechain-hp", type=float, default=20.0)
    comp_cmd.add_argument("--sidechain-lp", type=float, default=20000.0)
    comp_cmd.add_argument("--saturate-drive", type=float, default=0.0)
    comp_cmd.add_argument("--output-trim", type=float, default=0.0)
    comp_cmd.add_argument("--detector", type=str, default="RMS")
    comp_cmd.add_argument("--json", action="store_true")
    comp_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "compress":
        result = process_file(
            args.input, args.output,
            verbose=args.verbose,
            threshold_db=args.threshold,
            ratio=args.ratio,
            attack_ms=args.attack,
            release_ms=args.release,
            knee_db=args.knee,
            lookahead_ms=args.lookahead,
            hold_ms=args.hold,
            wet_dry_mix=args.wet_dry,
            stereo_link=args.stereo_link,
            sidechain_hp_hz=args.sidechain_hp,
            sidechain_lp_hz=args.sidechain_lp,
            saturate_drive_db=args.saturate_drive,
            output_trim_db=args.output_trim,
            detector_type=args.detector,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n==========================================")
            print(f"  FAURGE COMPRESSOR — PROCESSING REPORT")
            print(f"==========================================")
            print(f"  Status:           {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Processing time:  {result.get('processing_time_ms', 0):.2f} ms")
            print(f"  Input peak:       {result.get('input_peak_db', 0):.1f} dB")
            print(f"  Output peak:      {result.get('output_peak_db', 0):.1f} dB")
            print(f"  Input RMS:        {result.get('input_rms_db', 0):.1f} dB")
            print(f"  Output RMS:       {result.get('output_rms_db', 0):.1f} dB")
            print(f"  Gain reduction:   {result.get('gain_reduction_db', 0):.1f} dB")
            print(f"  Avg GR:           {result.get('avg_gain_reduction_db', 0):.1f} dB")
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
