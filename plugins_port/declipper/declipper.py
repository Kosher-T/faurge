"""
Portable Declipper — Faurge Portable Plugin
============================================

Single-file self-contained utility for Kaggle and local use.
Provides C++-accelerated declipping with a pure NumPy fallback.

Usage
-----
    import declipper_portable as dp

    # File-based API
    info = dp.clip_file("clean.wav", "degraded.wav", gain_db=8.0)
    result = dp.declip_file("degraded.wav", "restored.wav")

    # In-memory API
    audio_clipped = dp.clip(audio, gain_db=6.0)
    audio_restored, meta = dp.declip(audio_clipped, sr)

    # Quality assessment
    report = dp.quality_report(audio_clipped, audio_restored, sr)

    # Build C++ extension on Kaggle (one-time per session)
    dp.build_extension("/kaggle/input/datasets/itorousa/csrc-up/csrc")
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import typing as T
from pathlib import Path

import numpy as np

# ── C++ Backend ──────────────────────────────────────────────────────────────
_CPP_AVAILABLE = False
_cpp = None

try:
    import _faurge_declip_cpp as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    pass


# ── WAV I/O ──────────────────────────────────────────────────────────────────

def read_wav(path: str) -> T.Tuple[np.ndarray, int]:
    """Read WAV file, return (float32 array -1..1, sample_rate).

    Handles mono and multi-channel. Returns shape (N,) for mono, (N, C) otherwise.
    """
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


# ── Pure NumPy Fallback ──────────────────────────────────────────────────────

def _detect_hard_clips(
    audio: np.ndarray,
    threshold: float = 0.9999,
    min_len: int = 2,
) -> list:
    """Detect hard-clip regions where |sample| >= threshold.

    Returns list of (start, end, polarity). polarity = 1 for positive, -1 for negative.
    """
    n = len(audio)
    above = np.where(np.abs(audio) >= threshold)[0]
    if len(above) == 0:
        return []

    gaps = np.diff(above)
    breaks = np.where(gaps > 1)[0]
    segments = np.split(above, breaks + 1)

    regions = []
    for seg in segments:
        if len(seg) < min_len:
            continue
        start = int(seg[0])
        end = int(seg[-1])
        polarity = 1 if audio[start] >= 0 else -1
        regions.append((start, end, polarity))
    return regions


def _detect_soft_clips(
    audio: np.ndarray,
    threshold: float = 0.9999,
    deriv_thr: float = 0.5,
    min_len: int = 2,
) -> list:
    """Detect soft-clip regions via second-derivative flatness."""
    n = len(audio)
    if n < 4:
        return []

    high_amp = threshold * 0.9
    above = np.where(np.abs(audio) > high_amp)[0]
    if len(above) < 3:
        return []

    # Second derivative via convolution
    d2 = np.convolve(audio, [1.0, -2.0, 1.0], mode='valid')
    pad = 1
    d2_padded = np.zeros(n)
    d2_padded[pad:pad + len(d2)] = np.abs(d2)

    abs_audio = np.abs(audio)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_curv = np.where(abs_audio > 1e-6, d2_padded / abs_audio, d2_padded)

    flat_mask = norm_curv < (1.0 - deriv_thr) * 0.1
    high_mask = abs_audio > high_amp
    combined = flat_mask & high_mask

    transitions = np.diff(np.concatenate(([0], combined.astype(int), [0])))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    regions = []
    for s, e in zip(starts, ends):
        length = e - s
        if length < min_len:
            continue
        mid = (s + e) // 2
        polarity = 1 if audio[mid] >= 0 else -1
        regions.append((int(s), int(e - 1), polarity))
    return regions


def _merge_regions(
    regions: list,
    merge_gap: int = 3,
) -> list:
    """Merge overlapping or closely-spaced regions."""
    if not regions:
        return []
    sorted_reg = sorted(regions, key=lambda r: r[0])
    merged = [sorted_reg[0]]
    for cur in sorted_reg[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1] + 1 + merge_gap:
            merged[-1] = (prev[0], max(prev[1], cur[1]), prev[2])
        else:
            merged.append(cur)
    return merged


def _classify_severity(length: int) -> str:
    if length <= 4:
        return "Mild"
    if length <= 16:
        return "Moderate"
    if length <= 64:
        return "Severe"
    return "Critical"


def _hermite_reconstruct(audio: np.ndarray, start: int, end: int,
                         anchors_before: np.ndarray, anchors_after: np.ndarray,
                         peak_est: float, polarity: int) -> None:
    """Cubic Hermite interpolation for short clips (<=16 samples)."""
    clip_len = end - start + 1

    if len(anchors_before) < 2 or len(anchors_after) < 2:
        p0 = anchors_before[-1] if len(anchors_before) > 0 else 0.0
        p1 = anchors_after[0] if len(anchors_after) > 0 else 0.0
        t = np.linspace(1 / (clip_len + 1), clip_len / (clip_len + 1), clip_len)
        vals = p0 + t * (p1 - p0)
        audio[start:end + 1] = vals
        return

    p0 = anchors_before[-1]
    p1 = anchors_after[0]

    nb = len(anchors_before)
    if nb >= 3:
        m0 = (anchors_before[-1] - anchors_before[-3]) / 2.0
    else:
        m0 = anchors_before[-1] - anchors_before[-2]

    na = len(anchors_after)
    if na >= 3:
        m1 = (anchors_after[2] - anchors_after[0]) / 2.0
    else:
        m1 = anchors_after[1] - anchors_after[0]

    interval = clip_len + 1
    m0 *= interval
    m1 *= interval

    t = np.linspace(1 / interval, clip_len / interval, clip_len)
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    vals = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    if polarity == 1:
        vals = np.minimum(vals, peak_est)
    else:
        vals = np.maximum(vals, -peak_est)

    audio[start:end + 1] = vals


def _akima_reconstruct(audio: np.ndarray, start: int, end: int,
                       anchors_before: np.ndarray, anchors_after: np.ndarray,
                       peak_est: float, polarity: int) -> None:
    """Akima sub-spline for medium clips (17-64 samples)."""
    clip_len = end - start + 1
    nb = len(anchors_before)
    na = len(anchors_after)

    p0 = anchors_before[-1] if nb > 0 else audio[start]
    p1 = anchors_after[0] if na > 0 else audio[end]

    knots = np.concatenate([
        anchors_before,
        [p0, p1],
        anchors_after,
    ])
    nk = len(knots)

    xpos = np.concatenate([
        np.arange(-nb, 0, dtype=np.float32),
        [0.0, float(clip_len + 1)],
        np.arange(clip_len + 2, clip_len + 2 + na, dtype=np.float32),
    ])

    dx = np.diff(xpos)
    dd = np.where(dx != 0, np.diff(knots) / dx, 0.0)

    slopes = np.zeros(nk)
    for i in range(nk):
        if i < 2 or i >= nk - 2:
            slopes[i] = dd[min(i, nk - 2)] if i < nk - 1 else dd[nk - 2]
        else:
            w1 = abs(dd[i] - dd[i - 1])
            w2 = abs(dd[i - 2] - dd[i - 1])
            tw = w1 + w2
            if tw < 1e-12:
                slopes[i] = 0.5 * (dd[i - 1] + dd[i])
            else:
                wa = abs(dd[min(i, nk - 2)] - dd[i - 1])
                wb = abs(dd[i - 2] - dd[max(i - 3, 0)])
                tw2 = wa + wb
                if tw2 < 1e-12:
                    slopes[i] = 0.5 * (dd[i - 1] + dd[min(i, nk - 2)])
                else:
                    slopes[i] = (wa * dd[i - 1] + wb * dd[min(i, nk - 2)]) / tw2

    x0, x1 = xpos[nb], xpos[nb + 1]
    dx_seg = x1 - x0
    s0 = slopes[nb] * dx_seg
    s1 = slopes[nb + 1] * dx_seg

    x = np.arange(1, clip_len + 1, dtype=np.float32)
    t = (x - x0) / dx_seg
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    vals = h00 * p0 + h10 * s0 + h01 * p1 + h11 * s1

    if polarity == 1:
        vals = np.minimum(vals, peak_est)
    else:
        vals = np.maximum(vals, -peak_est)

    audio[start:end + 1] = vals


def _burg_ar_reconstruct(audio: np.ndarray, start: int, end: int,
                         order: int = 14,
                         peak_est: float = 1.15,
                         polarity: int = 1) -> None:
    """Burg AR extrapolation for long clips (>64 samples)."""
    clip_len = end - start + 1
    context_len = order * 4

    ctx_start = max(0, start - context_len)
    fwd_context = audio[ctx_start:start].copy()

    ctx_end = min(len(audio), end + 1 + context_len)
    bwd_context = audio[end + 1:ctx_end].copy()

    def burg_fit(data: np.ndarray, p: int) -> np.ndarray:
        n = len(data)
        if n <= p:
            return np.zeros(p, dtype=np.float32)
        a = np.zeros(p + 1, dtype=np.float32)
        a[0] = 1.0
        ef = data.copy().astype(np.float64)
        eb = data.copy().astype(np.float64)
        for m in range(p):
            num = np.sum(ef[m + 1:n] * eb[m:n - 1])
            den = np.sum(ef[m + 1:n] ** 2 + eb[m:n - 1] ** 2)
            km = -2.0 * num / den if den > 1e-30 else 0.0
            a_new = np.zeros(p + 1, dtype=np.float64)
            a_new[0] = 1.0
            a_new[1:m + 1] = a[1:m + 1] + km * a[m:0:-1]
            a_new[m + 1] = km
            a = a_new
            ef_new = np.zeros(n, dtype=np.float64)
            ef_new[m + 1:n] = ef[m + 1:n] + km * eb[m:n - 1]
            eb[m + 1:n] = eb[m:n - 1] + km * ef[m + 1:n]
            ef = ef_new
        return a[1:p + 1].astype(np.float32)

    fwd_coeffs = burg_fit(fwd_context, order)
    bwd_rev = burg_fit(bwd_context[::-1], order)

    fwd_pred = np.zeros(clip_len, dtype=np.float32)
    buf = list(fwd_context)
    for i in range(clip_len):
        val = 0.0
        b_len = len(buf)
        for k in range(order):
            if k < b_len:
                val -= fwd_coeffs[k] * buf[b_len - 1 - k]
        fwd_pred[i] = val
        buf.append(val)

    bwd_pred = np.zeros(clip_len, dtype=np.float32)
    buf = list(bwd_context[::-1])
    for i in range(clip_len):
        val = 0.0
        b_len = len(buf)
        for k in range(order):
            if k < b_len:
                val -= bwd_rev[k] * buf[b_len - 1 - k]
        bwd_pred[i] = val
        buf.append(val)
    bwd_pred = bwd_pred[::-1]

    t = np.linspace(0, 1, clip_len)
    w_fwd = 0.5 * (1.0 + np.cos(t * np.pi))
    w_bwd = 1.0 - w_fwd
    blended = w_fwd * fwd_pred + w_bwd * bwd_pred

    if polarity == 1:
        blended = np.minimum(blended, peak_est)
    else:
        blended = np.maximum(blended, -peak_est)

    audio[start:end + 1] = blended


def _estimate_peak(anchors_before: np.ndarray, anchors_after: np.ndarray,
                   clip_len: int, threshold: float,
                   peak_overshoot: float) -> float:
    slope_before = 0.0
    if len(anchors_before) >= 2:
        slope_before = anchors_before[-1] - anchors_before[-2]
    slope_after = 0.0
    if len(anchors_after) >= 2:
        slope_after = anchors_after[1] - anchors_after[0]
    avg_slope = (abs(slope_before) + abs(slope_after)) / 2.0
    raw_peak = threshold + avg_slope * clip_len * 0.25
    max_peak = threshold * peak_overshoot
    return min(raw_peak, max_peak)


def _crossfade_blend(audio: np.ndarray, start: int, end: int,
                     width: int = 8) -> None:
    if width <= 0:
        return

    fade_start = max(0, start - width)
    fade_len = start - fade_start
    if fade_len > 0:
        t = np.linspace(0, 1, fade_len)
        w = 0.5 * (1.0 - np.cos(np.pi * t))
        target = audio[start]
        idx = np.arange(fade_start, start)
        audio[idx] = audio[idx] * (1.0 - w * 0.1) + target * (w * 0.1)

    fade_start = end + 1
    fade_end = min(len(audio), fade_start + width)
    fade_len = fade_end - fade_start
    if fade_len > 0:
        t = np.linspace(0, 1, fade_len)
        w = 0.5 * (1.0 + np.cos(np.pi * t))
        target = audio[end]
        idx = np.arange(fade_start, fade_end)
        audio[idx] = audio[idx] * (1.0 - w * 0.1) + target * (w * 0.1)


def _anti_alias_filter(audio: np.ndarray, start: int, end: int,
                       sample_rate: int) -> None:
    if start >= end:
        return
    from scipy.signal import butter, filtfilt
    fc = sample_rate / 4.0
    if fc <= 0:
        return
    b, a = butter(2, fc / (sample_rate / 2), btype='low')
    seg = audio[start:end + 1]
    if len(seg) < 4:
        return
    filtered = filtfilt(b, a, seg)
    audio[start:end + 1] = filtered.astype(np.float32)


def _dc_block(audio: np.ndarray, start: int, end: int,
              fc: float = 10.0, sample_rate: int = 48000) -> None:
    if start >= end:
        return
    alpha = 1.0 - (2.0 * np.pi * fc / sample_rate)
    alpha = max(0.0, min(alpha, 0.9999))
    x = audio[start:end + 1].copy()
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + x[i] - x[i - 1]
    audio[start:end + 1] = y


def _compute_thdn(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate THD+N in dB using Hann-windowed DFT."""
    n = len(audio)
    if n < 64:
        return 0.0

    fft_size = 1
    while fft_size * 2 <= n and fft_size < 4096:
        fft_size *= 2

    offset = (n - fft_size) // 2
    window = np.hanning(fft_size)
    segment = audio[offset:offset + fft_size] * window

    spectrum = np.fft.rfft(segment)
    mag = np.abs(spectrum) ** 2
    half = len(mag)

    fund_bin = 1 + np.argmax(mag[1:])
    total_power = np.sum(mag[1:])
    fund_start = max(1, fund_bin - 2)
    fund_end = min(half, fund_bin + 3)
    fund_power = np.sum(mag[fund_start:fund_end])
    thdn_power = total_power - fund_power

    if fund_power < 1e-30:
        return -120.0
    return float(10.0 * np.log10(thdn_power / fund_power))


def _declip_numpy(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.9999,
    min_clip_length: int = 2,
    merge_gap: int = 3,
    anchor_size: int = 20,
    detect_soft_clip: bool = True,
    soft_clip_deriv_thr: float = 0.5,
    hermite_max_len: int = 16,
    akima_max_len: int = 64,
    ar_model_order: int = 14,
    peak_overshoot: float = 1.15,
    crossfade_width: int = 8,
    enable_anti_alias: bool = True,
    dc_block_freq_hz: float = 10.0,
) -> T.Tuple[np.ndarray, dict]:
    """Pure NumPy declipping pipeline. Returns (processed_audio, result_dict)."""
    t0 = time.perf_counter()
    out = audio.copy()
    n = len(out)

    if n == 0:
        return out, {
            "success": False, "error_message": "Empty audio buffer",
            "processing_time_ms": 0.0, "clip_report": {},
        }

    before = out.copy()

    # Stage 1: Detection
    hard = _detect_hard_clips(out, threshold, min_clip_length)
    soft = _detect_soft_clips(out, threshold, soft_clip_deriv_thr, min_clip_length) if detect_soft_clip else []
    all_regions = hard + soft
    all_regions = _merge_regions(all_regions, merge_gap)

    if not all_regions:
        t1 = time.perf_counter()
        return out, {
            "success": True, "error_message": "No clipping detected",
            "processing_time_ms": (t1 - t0) * 1000,
            "clip_report": {"num_regions": 0, "total_clipped_samples": 0, "percent_clipped": 0},
        }

    # Stage 2+3: Reconstruction
    for start_s, end_s, polarity in all_regions:
        clip_len = end_s - start_s + 1
        left_anchor_start = max(0, start_s - anchor_size)
        right_anchor_end = min(n, end_s + 1 + anchor_size)
        anchors_before = out[left_anchor_start:start_s]
        anchors_after = out[end_s + 1:right_anchor_end]

        peak_est = _estimate_peak(anchors_before, anchors_after, clip_len,
                                  threshold, peak_overshoot)

        if clip_len <= hermite_max_len:
            _hermite_reconstruct(out, start_s, end_s, anchors_before,
                                 anchors_after, peak_est, polarity)
        elif clip_len <= akima_max_len:
            _akima_reconstruct(out, start_s, end_s, anchors_before,
                               anchors_after, peak_est, polarity)
        else:
            _burg_ar_reconstruct(out, start_s, end_s, ar_model_order,
                                 peak_est, polarity)

    # Stage 4: Post-processing
    margin = crossfade_width
    for start_s, end_s, _ in all_regions:
        _crossfade_blend(out, start_s, end_s, crossfade_width)
        pf_start = max(0, start_s - margin)
        pf_end = min(n - 1, end_s + margin)
        if enable_anti_alias:
            _anti_alias_filter(out, pf_start, pf_end, sample_rate)
        _dc_block(out, pf_start, pf_end, dc_block_freq_hz, sample_rate)

    # Stage 5: Metrics
    total_clipped = sum(e - s + 1 for s, e, _ in all_regions)

    t1 = time.perf_counter()

    result = {
        "success": True,
        "processing_time_ms": (t1 - t0) * 1000,
        "before_thdn_db": _compute_thdn(before, sample_rate),
        "after_thdn_db": _compute_thdn(out, sample_rate),
        "clip_report": {
            "num_regions": len(all_regions),
            "total_clipped_samples": int(total_clipped),
            "percent_clipped": float(100.0 * total_clipped / n) if n > 0 else 0.0,
            "regions": [
                {
                    "index": i,
                    "start": int(s),
                    "end": int(e),
                    "length": int(e - s + 1),
                    "severity": _classify_severity(e - s + 1),
                }
                for i, (s, e, _) in enumerate(all_regions)
            ],
        },
    }

    return out, result


# ── Public API ───────────────────────────────────────────────────────────────

def clip(
    audio: np.ndarray,
    gain_db: float = 6.0,
    mode: str = "hard",
) -> np.ndarray:
    """Apply clipping distortion to audio.

    Parameters
    ----------
    audio : np.ndarray
        Input audio, float32 in [-1, 1].
    gain_db : float
        Gain applied before clipping (dB).
    mode : str
        "hard" — clamp to [-1, 1] after gain.
        "soft" — tanh(gain * audio).
        "random_segment" — clip random segments (see clip_file).

    Returns
    -------
    np.ndarray
        Clipped audio.
    """
    out = audio.copy().astype(np.float32)
    gain_lin = 10.0 ** (gain_db / 20.0)

    if mode == "hard":
        out = out * gain_lin
        out = np.clip(out, -1.0, 1.0)
    elif mode == "soft":
        out = np.tanh(out * gain_lin)
    elif mode == "random_segment":
        rng = np.random.default_rng(42)
        seg_size = 1024
        n = len(out)
        for i in range(0, n, seg_size):
            if rng.random() < 0.3:
                end = min(i + seg_size, n)
                seg = out[i:end] * gain_lin
                out[i:end] = np.clip(seg, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown clip mode: {mode}")

    return out


def declip(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.9999,
    ar_order: int = 14,
    verbose: bool = False,
    **kwargs,
) -> T.Tuple[np.ndarray, dict]:
    """Restore clipped audio.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio in [-1, 1]. Processed in-place.
    sample_rate : int
        Sample rate in Hz.
    threshold, ar_order : float, int
        Clipping threshold and AR model order.
    verbose : bool
        Print per-region info to stderr.
    **kwargs : passed to C++ or NumPy backend.

    Returns
    -------
    (processed_audio, result_dict)
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    n = len(audio)

    if n == 0:
        return audio, {"success": False, "error_message": "Empty buffer"}

    if audio.ndim == 1:
        result = _run_single_channel(audio, sample_rate, threshold, ar_order, verbose, kwargs)
        return audio, result
    else:
        results = []
        for ch in range(audio.shape[1]):
            ch_audio = audio[:, ch]
            ch_result = _run_single_channel(ch_audio, sample_rate, threshold, ar_order, verbose, kwargs)
            results.append(ch_result)
        return audio, results[-1]


def _try_import_cpp():
    global _cpp, _CPP_AVAILABLE
    if not _CPP_AVAILABLE:
        try:
            import _faurge_declip_cpp as _cpp_mod
            _cpp = _cpp_mod
            _CPP_AVAILABLE = True
        except ImportError:
            pass
    return _CPP_AVAILABLE


def _run_single_channel(audio, sr, threshold, ar_order, verbose, kwargs):
    if _try_import_cpp():
        result = _cpp.declip(
            audio, sr,
            clip_threshold=threshold,
            ar_model_order=ar_order,
            **kwargs,
        )
        if verbose:
            report = result.get("clip_report", {})
            print(f"[declipper] C++: {report.get('num_regions', 0)} regions, "
                  f"{report.get('percent_clipped', 0):.1f}% clipped",
                  file=sys.stderr)
        return result
    else:
        out, result = _declip_numpy(
            audio, sr,
            threshold=threshold,
            ar_model_order=ar_order,
            **kwargs,
        )
        audio[:] = out
        if verbose:
            report = result.get("clip_report", {})
            print(f"[declipper] NumPy: {report.get('num_regions', 0)} regions, "
                  f"{report.get('percent_clipped', 0):.1f}% clipped",
                  file=sys.stderr)
        return result


def clip_file(
    input_path: str,
    output_path: str,
    gain_db: float = 6.0,
    mode: str = "hard",
) -> dict:
    """Read WAV, clip it, write WAV. Returns metadata."""
    audio, sr = read_wav(input_path)
    was_mono = audio.ndim == 1

    if audio.ndim == 1:
        audio = clip(audio, gain_db, mode)
    else:
        for ch in range(audio.shape[1]):
            audio[:, ch] = clip(audio[:, ch], gain_db, mode)

    write_wav(output_path, audio, sr)
    peak = float(np.max(np.abs(audio)))
    return {
        "input": input_path,
        "output": output_path,
        "gain_db": gain_db,
        "mode": mode,
        "sample_rate": sr,
        "peak": peak,
        "clipped": peak >= 0.9999,
    }


def declip_file(
    input_path: str,
    output_path: str,
    threshold: float = 0.9999,
    ar_order: int = 14,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Read WAV, declip it, write WAV. Returns result dict."""
    audio, sr = read_wav(input_path)
    was_mono = audio.ndim == 1

    if audio.ndim == 1:
        audio, result = declip(audio, sr, threshold, ar_order, verbose, **kwargs)
    else:
        result = None
        for ch in range(audio.shape[1]):
            ch_arr = audio[:, ch]
            ch_arr, ch_result = declip(ch_arr, sr, threshold, ar_order, verbose, **kwargs)
            if result is None:
                result = ch_result

    write_wav(output_path, audio, sr)
    result["input"] = input_path
    result["output"] = output_path
    return result


def quality_report(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
) -> dict:
    """One-shot quality assessment. No loop.

    Returns dict with THD+N before/after, SNR, peak/RMS levels.
    """
    orig = np.asarray(original, dtype=np.float32)
    proc = np.asarray(processed, dtype=np.float32)

    if orig.ndim == 2:
        orig = orig.mean(axis=1)
        proc = proc.mean(axis=1)

    min_len = min(len(orig), len(proc))
    orig = orig[:min_len]
    proc = proc[:min_len]

    noise = proc - orig
    sig_power = np.mean(proc ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = float(10.0 * np.log10(sig_power / noise_power)) if noise_power > 1e-30 else float('inf')

    orig_thdn = _compute_thdn(orig, sample_rate)
    proc_thdn = _compute_thdn(proc, sample_rate)

    return {
        "snr_db": snr_db,
        "thdn_before_db": orig_thdn,
        "thdn_after_db": proc_thdn,
        "peak_before": float(np.max(np.abs(orig))),
        "peak_after": float(np.max(np.abs(proc))),
        "rms_before": float(np.sqrt(np.mean(orig ** 2))),
        "rms_after": float(np.sqrt(np.mean(proc ** 2))),
        "improvement_db": orig_thdn - proc_thdn,
    }


def build_extension(
    source_dir: str = "csrc",
    build_dir: str = "/tmp/declipper_build",
    force: bool = False,
) -> bool:
    """Compile the C++ extension.

    Parameters
    ----------
    source_dir : str
        Path to csrc/ directory containing CMakeLists.txt.
    build_dir : str
        Build directory.
    force : bool
        Rebuild even if already built.

    Returns
    -------
    bool
        True if build succeeded, False otherwise.
    """
    lib_name = f"_faurge_declip_cpp{'.cpython-312-x86_64-linux-gnu' if sys.platform == 'linux' else ''}.so"

    if not force:
        try:
            import _faurge_declip_cpp
            return True
        except ImportError:
            pass

    print(f"[declipper] Building C++ extension from {source_dir}...", file=sys.stderr)

    try:
        subprocess.run(
            ["pip", "install", "-q", "pybind11"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        print("[declipper] Warning: pip install pybind11 failed; trying apt", file=sys.stderr)
        try:
            subprocess.run(
                ["apt-get", "install", "-y", "-qq", "pybind11-dev"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            print("[declipper] Error: cannot install pybind11", file=sys.stderr)
            return False

    os.makedirs(build_dir, exist_ok=True)
    try:
        cmake_pybind11_dir = None
        try:
            import pybind11
            cmake_pybind11_dir = pybind11.get_cmake_dir()
        except ImportError:
            pass

        cmake_cmd = ["cmake", source_dir, "-DCMAKE_BUILD_TYPE=Release"]
        if cmake_pybind11_dir:
            cmake_cmd.append(f"-Dpybind11_DIR={cmake_pybind11_dir}")

        subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True)
        subprocess.run(["make", "-j$(nproc)"], cwd=build_dir, check=True, capture_output=True, shell=True)

        for f in os.listdir(build_dir):
            if f.startswith("_faurge_declip_cpp") and f.endswith(".so"):
                dest = os.path.join(os.path.dirname(__file__), f)
                subprocess.run(["cp", os.path.join(build_dir, f), dest], check=True)
                print(f"[declipper] Copied {f} to {dest}", file=sys.stderr)
                break
        else:
            print("[declipper] Error: .so not found after build", file=sys.stderr)
            return False

        import importlib
        importlib.invalidate_caches()
        if _try_import_cpp():
            print("[declipper] C++ extension loaded.", file=sys.stderr)
        else:
            print("[declipper] Warning: .so copied but import failed; "
                  "restart kernel and re-run build.", file=sys.stderr)

        print("[declipper] Build complete.", file=sys.stderr)
        return True

    except subprocess.CalledProcessError as e:
        print(f"[declipper] Build failed: {e}", file=sys.stderr)
        print(e.stderr.decode() if e.stderr else "", file=sys.stderr)
        return False


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """Command-line interface for local testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Portable Declipper")
    sub = parser.add_subparsers(dest="command")

    clip_cmd = sub.add_parser("clip", help="Clip audio")
    clip_cmd.add_argument("input")
    clip_cmd.add_argument("output")
    clip_cmd.add_argument("--gain-db", type=float, default=6.0)
    clip_cmd.add_argument("--mode", choices=["hard", "soft"], default="hard")

    declip_cmd = sub.add_parser("declip", help="Declip audio")
    declip_cmd.add_argument("input")
    declip_cmd.add_argument("output")
    declip_cmd.add_argument("--threshold", type=float, default=0.9999)
    declip_cmd.add_argument("--ar-order", type=int, default=14)
    declip_cmd.add_argument("--json", action="store_true")
    declip_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original")
    quality_cmd.add_argument("processed")

    build_cmd = sub.add_parser("build", help="Build C++ extension")
    build_cmd.add_argument("--source-dir", default="csrc")
    build_cmd.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "clip":
        info = clip_file(args.input, args.output, args.gain_db, args.mode)
        print(json.dumps(info, indent=2))

    elif args.command == "declip":
        result = declip_file(args.input, args.output, args.threshold, args.ar_order, args.verbose)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            report = result.get("clip_report", {})
            print(f"Success: {result.get('success')}")
            print(f"Regions: {report.get('num_regions', 0)}")
            print(f"Clipped: {report.get('percent_clipped', 0):.1f}%")
            print(f"Before THD+N: {result.get('before_thdn_db', 0):.1f} dB")
            print(f"After THD+N:  {result.get('after_thdn_db', 0):.1f} dB")

    elif args.command == "quality":
        orig, sr1 = read_wav(args.original)
        proc, sr2 = read_wav(args.processed)
        report = quality_report(orig, proc, sr1)
        print(json.dumps(report, indent=2))

    elif args.command == "build":
        ok = build_extension(args.source_dir, force=args.force)
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
