# Portable DFN3 — Faurge

A self-contained DeepFilterNet3 speech denoising plugin designed to run inside **Kaggle Utility Scripts** for large-scale training data generation. Single file. Install deps and go.

---

## Why This Exists

### The Problem

We are training two neural networks — **Ursula** and **Genesis** — to restore degraded audio in real time. These models need hundreds of thousands of clean→degraded audio pairs for training. The degradation pipeline must:

1. **Add noise** to pristine audio to create realistic "before" states (via our augmentation pipeline)
2. **Denoise** the same audio to compute training targets (what the models should learn to do)
3. Run at **GPU speed** because thousands of files means hours of audio
4. Execute on **Kaggle**, where datasets live and GPU quota is allocated

### The Constraints

| Constraint | Implication |
|---|---|
| **Kaggle Utility Scripts are single `.py` files** | You cannot rely on multi-file modules. Everything must live in one file that gets copied to `/kaggle/lib/user/`. |
| **DeepFilterNet3 is a neural network** | Unlike DSP-based plugins (declipper/exciter), DFN3 requires PyTorch inference and a model download. No pure-Python fallback exists. |
| **Kaggle has PyTorch pre-installed** | No PyTorch build step needed. `pip install deepfilternet` is sufficient. |
| **Kaggle has T4 GPUs** | The model runs ~50x faster on GPU. The plugin defaults to CUDA with automatic CPU fallback. |
| **The notebook has internet access** | Model weights (~8 MB) download automatically from GitHub on first `init_model()` call. No offline competition constraints. |
| **48 kHz input only** | DeepFilterNet3 is trained at 48 kHz. The plugin transparently resamples via `librosa.resample`. |

### The Use Case

```python
import dfn3                                                    # one .py file
_, _, _ = dfn3.init_model()                                    # download weights once

for clean_path in thousands_of_wavs:
    result = dfn3.denoise_file(f"degraded/{clean_path.name}",
                                f"restored/{clean_path.name}")
    # → Ursula trains on (degraded → restored) pairs
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Kaggle Notebook                       │
│  pip install -> torch deepfilternet librosa               │
│  import dfn3                                              │
│  dfn3.denoise_file("noisy.wav", "clean.wav")              │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  dfn3.py (single file)                     │
│  ┌─────────────────────────────────────────────────┐     │
│  │  WAV I/O           scipy.io.wavfile             │     │
│  │  Model init        df.enhance.init_df()         │     │
│  │  Denoise           df.enhance.enhance()          │     │
│  │  Resample          librosa.resample              │     │
│  │  CLI               argparse subcommands          │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│             DeepFilterNet3 (via pip deepfilternet)         │
│  ┌─────────────────────────────────────────────────┐     │
│  │  init_df() → downloads model (~8 MB) to cache   │     │
│  │  enhance()  → STFT → ERB → DfNet → ISTFT        │     │
│  │  Backend: pre-compiled Rust STFT (deepfilterlib) │     │
│  │  Model:   PyTorch, ~1.7M params, 40 ms latency  │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### One Backend, One API

When you call `denoise(audio, sr)`, the script:

1. **Resamples** to 48 kHz (if needed) via `librosa.resample`
2. **Calls** `init_df()` to download/lazily load the DeepFilterNet3 model
3. **Runs** `df.enhance.enhance()` which performs STFT → ERB feature extraction → DfNet model forward pass → ISTFT → time-domain denoised audio
4. **Resamples** back to the original sample rate (if needed)
5. **Returns** the denoised float32 array and a metadata dict

The model download (`init_model()`) is a one-time ~5s cost per notebook session. Subsequent calls use cached weights from `~/.cache/DeepFilterNet/`.

---

## The Denoising Pipeline

The core is the official DeepFilterNet3 model by Schröter et al. (INTERSPEECH 2023). Internally:

| Stage | What | Method |
|---|---|---|
| 1 | **STFT Analysis** | 960-pt FFT, 480-hop, 50% overlap, Hann window (via Rust `deepfilterlib`) |
| 2 | **ERB Feature Extraction** | 32-band gammatone-like filterbank mapped to ERB scale |
| 3 | **DfNet Model** | Dual-path: ERB mask estimation + deep filtering coefficient prediction. SqueezedGRU encoders, grouped convolutions, ~1.7M params |
| 4 | **Deep Filtering** | 5th-order complex FIR filters applied per frequency bin over time |
| 5 | **ISTFT Synthesis** | Overlap-add reconstruction via `deepfilterlib` |

Metrics (SNR, peak, RMS) are computed before/after for quality assessment.

---

## Files

```
plugins_port/dfn3/
├── dfn3.py              # ← THE Utility Script. Single file. Copy-paste to Kaggle.
└── test/
    ├── test_denoise.py   # 8 tests: WAV roundtrip, empty/short audio, quality fields
    └── test_quality.py   # 5 tests: perfect score, SNR comparison, stereo averaging
```

No `csrc/` directory — DFN3 is a neural network, not DSP. There is no C++ build step.

---

## Kaggle Workflow

### First-time setup (once per notebook session)

```python
import dfn3

# Download model weights — ~5 seconds, one-time per session
dfn3.init_model()
# Downloads DeepFilterNet3 from GitHub to ~/.cache/DeepFilterNet/DeepFilterNet3/
```

### Daily use

```python
# Denoise a single file
result = dfn3.denoise_file("noisy.wav", "clean.wav")
print(result["processing_time_ms"], "ms on", result["device"])

# Quality assessment (one-shot, no loop)
report = dfn3.quality_report(noisy, clean, sr)
# → {"snr_db": 18.4, "peak_before": 0.95, "peak_after": 0.85, ...}

# In-memory (avoid WAV I/O for batching)
audio, sr = dfn3.read_wav("file.wav")
denoised, meta = dfn3.denoise(audio, sr)
dfn3.write_wav("out.wav", denoised, sr)

# With post-filter for extra noise reduction
result = dfn3.denoise_file("noisy.wav", "clean.wav", pf=True)

# Force CPU
result = dfn3.denoise_file("noisy.wav", "clean.wav", device="cpu")

# Attenuation limit — only suppress up to 12 dB of noise
result = dfn3.denoise_file("noisy.wav", "clean.wav", attenu_lim_db=12)
```

### Batch processing

```python
import glob
import dfn3

dfn3.init_model()
paths = glob.glob("/kaggle/input/dataset/noisy/*.wav")

for p in paths:
    result = dfn3.denoise_file(p, f"/kaggle/working/clean/{p.name}")
```

---

## Local Development

```bash
cd plugins_port/dfn3

# Install dependencies
pip install torch deepfilternet librosa scipy

# Run tests
python3 test/test_denoise.py
python3 test/test_quality.py

# CLI
python3 dfn3.py download                         # download model weights
python3 dfn3.py denoise noisy.wav clean.wav      # denoise a file
python3 dfn3.py denoise noisy.wav clean.wav --pf # with post-filter
python3 dfn3.py quality orig.wav proc.wav        # quality report
python3 dfn3.py denoise in.wav out.wav --json    # JSON output
```

### Dependencies for local testing

- Python 3.10+, NumPy, SciPy, librosa, PyTorch ≥ 1.9
- `deepfilternet` (pip-installable, includes pre-compiled Rust `deepfilterlib`)

---

## API Reference

### `init_model(model_dir=None, device="auto", post_filter=False) -> (model, df_state, suffix)`
Download and cache the DeepFilterNet3 model. One-time build step per session.

### `denoise(audio, sample_rate, **kwargs) -> (np.ndarray, dict)`
Denoise audio. Returns cleaned array + metadata dict.

### `denoise_file(input_path, output_path, **kwargs) -> dict`
Read WAV → denoise → write WAV.

### `quality_report(original, processed, sr) -> dict`
One-shot: SNR, peak, RMS, improvement.

### `read_wav(path) -> (np.ndarray, int)`
Read WAV to float32 [-1, 1].

### `write_wav(path, audio, sr)`
Write float32 audio to WAV.

---

## Design Decisions

**Why DeepFilterNet3 instead of a simpler model?** DFN3 is state-of-the-art for real-time speech enhancement at 48 kHz (40 ms latency). Its dual-path architecture (ERB mask + deep filtering) produces clean, artifact-free output that works as a training target for Ursula.

**Why no pure-Python fallback?** Unlike the declipper (which has a complete NumPy fallback), DFN3 is a learned neural network with ~1.7M parameters. There is no algorithmic substitute. PyTorch is pre-installed on Kaggle and `deepfilternet` is a one-line pip install. No fallback is necessary.

**Why not ONNX Runtime?** The official `deepfilternet` package bundles a pre-compiled Rust STFT/ISTFT (`deepfilterlib`) that is faster and more robust than any Python-based STFT implementation. Using the pip package avoids reimplementing the feature extraction pipeline.

**Why is the quality check one-shot with no loop?** Same rationale as the declipper — for training data generation, a single SNR measurement is sufficient. The original "Fabian Orchestrator" reflection loop is for live inference, not dataset creation.
