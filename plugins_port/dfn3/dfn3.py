"""
Portable DFN3 — Faurge Portable Plugin
========================================

Single-file self-contained DeepFilterNet3 speech denoiser for Kaggle and local use.

Usage
-----
    import dfn3

    # Build step — download model weights (one-time per session)
    _, _, _ = dfn3.init_model()

    # File-based API
    result = dfn3.denoise_file("noisy.wav", "clean.wav")

    # In-memory API
    audio_clean, meta = dfn3.denoise(audio_noisy, sr)

    # Quality assessment
    report = dfn3.quality_report(audio_noisy, audio_clean, sr)
"""

import json
import os
import sys
import time
import typing as T

import numpy as np

_DF_AVAILABLE = False
_model = None
_df_state = None
_suffix = None

try:
    from df.enhance import enhance as _df_enhance, init_df as _df_init_df
    _DF_AVAILABLE = True
except ImportError:
    pass


def read_wav(path: str) -> T.Tuple[np.ndarray, int]:
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
    import scipy.io.wavfile as wavfile
    audio_clip = np.clip(audio, -1.0, 1.0)
    data = (audio_clip * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, data)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    channels = []
    for ch in range(audio.shape[0]):
        channels.append(librosa.resample(audio[ch], orig_sr=orig_sr, target_sr=target_sr))
    out = np.stack(channels, axis=0)
    if was_1d:
        out = out[0]
    return out


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _infer_device_from_model(model: T.Any) -> str:
    """Return the device string of a PyTorch model's parameters."""
    import torch
    try:
        p = next(model.parameters())
        return p.device.type
    except (StopIteration, RuntimeError):
        return "cpu"


def init_model(
    model_dir: T.Optional[str] = None,
    device: str = "auto",
    post_filter: bool = False,
) -> T.Tuple[T.Any, T.Any, str]:
    global _model, _df_state, _suffix

    if not _DF_AVAILABLE:
        raise ImportError(
            "deepfilternet is not installed. Run: pip install deepfilternet torch"
        )

    dev = _resolve_device(device)
    model, df_state, suffix, epoch = _df_init_df(
        model_base_dir=model_dir,
        post_filter=post_filter,
        log_level="WARNING",
        log_file=None,
    )
    model = model.to(dev)
    model.eval()

    _model = model
    _df_state = df_state
    _suffix = suffix

    return model, df_state, suffix


def denoise(
    audio: np.ndarray,
    sample_rate: int,
    model: T.Any = None,
    df_state: T.Any = None,
    device: str = "auto",
    post_filter: bool = False,
    attenuate_lim_db: T.Optional[float] = None,
) -> T.Tuple[np.ndarray, dict]:
    import torch
    t0 = time.perf_counter()

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    n = len(audio)

    if n == 0:
        return audio, {
            "success": False,
            "error_message": "Empty audio buffer",
            "processing_time_ms": 0.0,
        }

    orig_sr = sample_rate
    was_1d = audio.ndim == 1

    if model is None or df_state is None:
        if _model is None or _df_state is None:
            m, d, _ = init_model(device=device, post_filter=post_filter)
            if model is None:
                model = m
            if df_state is None:
                df_state = d
        else:
            if model is None:
                model = _model
            if df_state is None:
                df_state = _df_state

    dev = _resolve_device(device)
    if model is not None:
        dev = _infer_device_from_model(model)
    df_sr = 48000

    audio_48k = _resample(audio, orig_sr, df_sr)

    if was_1d:
        audio_t = torch.from_numpy(audio_48k).unsqueeze(0).to(dev)
    else:
        audio_t = torch.from_numpy(audio_48k.T).to(dev)

    with torch.no_grad():
        enhanced = _df_enhance(
            model, df_state, audio_t,
            pad=True,
            atten_lim_db=attenuate_lim_db,
        )
    enhanced = enhanced.cpu().numpy()

    if was_1d:
        enhanced = enhanced[0]
    else:
        enhanced = enhanced.T

    enhanced = np.clip(enhanced, -1.0, 1.0).astype(np.float32)

    if orig_sr != df_sr:
        enhanced = _resample(enhanced, df_sr, orig_sr)

    t1 = time.perf_counter()

    result = {
        "success": True,
        "processing_time_ms": (t1 - t0) * 1000,
        "model": _suffix or "DeepFilterNet3",
        "device": dev,
        "input_samples": int(n),
        "output_samples": int(len(enhanced) if was_1d else enhanced.shape[0]),
        "sample_rate": orig_sr,
    }

    return enhanced, result


def denoise_file(
    input_path: str,
    output_path: str,
    device: str = "auto",
    post_filter: bool = False,
    attenuate_lim_db: T.Optional[float] = None,
    verbose: bool = False,
) -> dict:
    audio, sr = read_wav(input_path)

    if audio.ndim == 1:
        enhanced, result = denoise(
            audio, sr,
            device=device,
            post_filter=post_filter,
            attenuate_lim_db=attenuate_lim_db,
        )
    else:
        result = None
        channels = []
        for ch in range(audio.shape[1]):
            ch_arr = audio[:, ch]
            ch_enh, ch_result = denoise(
                ch_arr, sr,
                device=device,
                post_filter=post_filter,
                attenuate_lim_db=attenuate_lim_db,
            )
            channels.append(ch_enh)
            if result is None:
                result = ch_result
        enhanced = np.column_stack(channels)

    write_wav(output_path, enhanced, sr)
    result["input"] = input_path
    result["output"] = output_path

    if verbose:
        ms = result.get("processing_time_ms", 0)
        print(f"[dfn3] Denoised {os.path.basename(input_path)} in {ms:.1f} ms "
              f"on {result.get('device', '?')}",
              file=sys.stderr)

    return result


def quality_report(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
) -> dict:
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

    before_power = np.mean(orig ** 2)
    after_power = np.mean(proc ** 2)
    improvement_db = float(10.0 * np.log10(after_power / before_power)) if before_power > 1e-30 else 0.0

    return {
        "snr_db": snr_db,
        "peak_before": float(np.max(np.abs(orig))),
        "peak_after": float(np.max(np.abs(proc))),
        "rms_before": float(np.sqrt(np.mean(orig ** 2))),
        "rms_after": float(np.sqrt(np.mean(proc ** 2))),
        "improvement_db": float(improvement_db),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Portable DFN3 — DeepFilterNet3 Denoiser")
    sub = parser.add_subparsers(dest="command")

    download_cmd = sub.add_parser("download", help="Download and cache model weights")
    download_cmd.add_argument("--model-dir", type=str, default=None)
    download_cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    denoise_cmd = sub.add_parser("denoise", help="Denoise audio file")
    denoise_cmd.add_argument("input", type=str)
    denoise_cmd.add_argument("output", type=str)
    denoise_cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    denoise_cmd.add_argument("--pf", action="store_true", help="Enable post-filter")
    denoise_cmd.add_argument("--atten-lim", type=float, default=None,
                             help="Attenuation limit in dB")
    denoise_cmd.add_argument("--json", action="store_true")
    denoise_cmd.add_argument("--verbose", action="store_true")

    quality_cmd = sub.add_parser("quality", help="Quality report")
    quality_cmd.add_argument("original", type=str)
    quality_cmd.add_argument("processed", type=str)

    args = parser.parse_args()

    if args.command == "download":
        model, df_state, suffix = init_model(
            model_dir=args.model_dir,
            device=args.device,
        )
        print(json.dumps({
            "success": True,
            "model": suffix,
            "device": next(model.parameters()).device.type,
        }, indent=2))

    elif args.command == "denoise":
        result = denoise_file(
            args.input, args.output,
            device=args.device,
            post_filter=args.pf,
            attenuate_lim_db=args.atten_lim,
            verbose=args.verbose,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Success: {result.get('success')}")
            print(f"Model: {result.get('model', '?')}")
            print(f"Device: {result.get('device', '?')}")
            print(f"Time: {result.get('processing_time_ms', 0):.1f} ms")

    elif args.command == "quality":
        orig, sr1 = read_wav(args.original)
        proc, sr2 = read_wav(args.processed)
        report = quality_report(orig, proc, sr1)
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
