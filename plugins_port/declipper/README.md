# Portable Declipper — Faurge

A self-contained, C++-accelerated audio declipping plugin designed to run inside **Kaggle Utility Scripts** for large-scale training data generation. Single file. No package manager. No environment setup. Upload it and go.

---

## Why This Exists

### The Problem

We are training two neural networks — **Ursula** and **Genesis** — to restore degraded audio in real time. These models need hundreds of thousands of clean→degraded audio pairs for training. The degradation pipeline must:

1. **Clip** pristine audio to create realistic "before" states (hard clipping, soft saturation, random-segment distortion)
2. **Declip** the same audio to compute training targets (what the models should learn to do)
3. Run at **C++ speed** because thousands of files means hours of audio
4. Execute on **Kaggle**, where datasets live and GPU quota is allocated

### The Constraints

| Constraint | Implication |
|---|---|
| **Kaggle Utility Scripts are single `.py` files** | You cannot `pip install` a package, upload a wheel, or rely on multi-file modules. Everything must live in one file that gets copied to `/kaggle/lib/user/`. |
| **No apt-get at runtime** (in internet-disabled competitions) | We cannot rely on `libsndfile` or system-level audio libraries. |
| **g++ and cmake are pre-installed** on Kaggle's Debian 13 Docker | C++ compilation at notebook startup is feasible — a one-time ~30s cost per session. |
| **pybind11 is not pre-installed** but pip-installable | The C++ extension needs a build step, but it only needs pybind11 (header-only, no system deps). |
| **Kaggle has Python 3.12 / Linux x86_64** | Pre-compiled wheels must match exactly. Compile-at-runtime avoids this fragility. |
| **25-second clips at most (typically 7s)** | At 48 kHz that's 1.2M samples max with maybe 8–12 dB of clipping. The Burg AR extrapolation (the most complex tier) is needed for the longer clip regions that heavy gain creates. |
| **The notebook must work offline** | If the C++ build fails or internet is disabled, the script must still work — pure NumPy fallback. |

### The Use Case

```python
import declipper as dp                                    # one .py file
dp.build_extension("/kaggle/input/datasets/itorousa/csrc-up/csrc")  # compile once

for clean_path in thousands_of_wavs:
    dp.clip_file(clean_path, f"degraded/{clean_path.name}", gain_db=8.0)
    dp.declip_file(f"degraded/{clean_path.name}", f"restored/{clean_path.name}")
    # → Ursula trains on (degraded → restored) pairs
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Kaggle Notebook                      │
│  import declipper as dp                                  │
│  dp.declip_file("in.wav", "out.wav")                     │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  declipper.py (single file)               │
│  ┌─────────────────────────────────────────────────┐     │
│  │  WAV I/O           scipy.io.wavfile             │     │
│  │  C++ bridge        _faurge_declip_cpp (.so)     │     │
│  │  NumPy fallback    full 5-stage pipeline        │     │
│  │  Build helper      cmake + pip install pybind11 │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────┘
                       │ (calls into .so or NumPy)
┌──────────────────────▼───────────────────────────────────┐
│               C++ Core (csrc/)                            │
│  ┌─────────────────────────────────────────────────┐     │
│  │  pybind11 bindings    (bindings.cpp)             │     │
│  │  faurge::Declipper    (orchestrator)             │     │
│  │  faurge::ClipDetector (detection)                │     │
│  │  faurge::Reconstructor (Hermite/Akima/AR)        │     │
│  │  faurge::PostFilter   (crossfade/AA/DC block)   │     │
│  │  faurge::Metrics      (THD+N / SNR)             │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Two Backends, One API

When you call `declip(audio, sr)`, the script checks for the C++ extension in this order:

1. **C++ (pybind11)** — Look for `_faurge_declip_cpp.so` on `sys.path`. If found, call `_cpp.declip(audio, sr, ...)` which modifies the NumPy array in-place and returns a result dict. **~5ms for 60s of audio at 48 kHz.**

2. **Pure NumPy fallback** — If the `.so` is not available, run the full 5-stage pipeline entirely in NumPy. Same algorithm, same regions, ~200ms for the same audio. **Always works, no dependencies beyond numpy/scipy.**

The build helper `build_extension()` compiles the `.so` at runtime and updates the module globals so subsequent calls use C++.

---

## The Declipping Algorithm (5 Stages)

The core is ported directly from `plugins/declipper/` (C++17, no libsndfile, no external deps). The algorithm adapts to the severity of the damage:

| Stage | What | Clip Length | Method |
|---|---|---|---|
| 1 | **Detection** | — | Hard-clip: `|sample| >= 0.9999`. Soft-clip: second-derivative flatness while amplitude is high. Regions are merged if within 3 samples of each other. |
| 2 | **Hermite Spline** | ≤16 samples | Cubic Hermite interpolation with finite-difference slope estimation. C1-continuous at boundaries. Guarantees no overshoot. |
| 3 | **Akima Sub-Spline** | 17–64 samples | Piecewise cubic with Akima's weighted-slope formula. Avoids the Runge oscillation that natural splines exhibit on longer gaps. |
| 4 | **Burg AR** | >64 samples | Autoregressive extrapolation using Burg's method (order 14). Forward-predicts from pre-clip context, backward-predicts from post-clip context, blends with raised-cosine crossfade. Stable for sustained clip regions. |
| 5 | **Post-processing** | — | Raised-cosine crossfade at region boundaries. 2nd-order Butterworth LPF (zero-phase, forward+backward). 1st-order DC blocker at 10 Hz. |

Metrics (THD+N via Hann-windowed DFT, per-region SNR) are computed before and after for quality assessment.

---

## Files

```
plugins_port/declipper/
├── declipper.py              # ← THE Utility Script. Single file. Copy-paste to Kaggle.
├── test/
│   ├── test_clip.py          # 7 tests: hard/soft clip, gain, passthrough, stereo
│   ├── test_declip.py        # 9 tests: clean passthrough, SNR improvement, edge cases
│   └── test_quality.py       # 6 tests: THD+N, field completeness, stereo averaging
├── csrc/                     # C++ source — upload this as a Kaggle Dataset
│   ├── CMakeLists.txt        # pybind11 build (no libsndfile, no system deps)
│   ├── bindings.cpp          # pybind11 module exposing declip() + clip()
│   ├── include/faurge/       # Headers for all 5 stages
│   └── src/                  # C++ implementations
└── README.md
```

### Why `csrc/` is separate

The `declipper.py` is one self-contained file you add as a Kaggle Utility Script. The `csrc/` directory is a **separate Kaggle Dataset** because:
- Utility Scripts have a soft file size limit and shouldn't contain C++ source
- The C++ build is optional — the script works without it
- You can update the C++ code without touching the script

On Kaggle:
```
Utility Script:  /kaggle/lib/user/declipper.py
Dataset (csrc):  /kaggle/input/datasets/itorousa/csrc-up/
```

---

## Kaggle Workflow

### First-time setup (once per notebook session)

```python
import declipper as dp

# Compile C++ extension — 30 seconds, one-time per session
dp.build_extension("/kaggle/input/datasets/itorousa/csrc-up/csrc")
# Installs pybind11 → cmake → make → copies .so → re-imports
```

### Daily use

```python
# Clip pristine audio
dp.clip_file("source.wav", "degraded.wav", gain_db=8.0, mode="hard")

# Declip for training target
result = dp.declip_file("degraded.wav", "restored.wav")

# Quality assessment (one-shot, no loop)
report = dp.quality_report(clean, restored, sr)
# → {"snr_db": 12.3, "thdn_before_db": -8.5, "thdn_after_db": -15.2, ...}

# In-memory (avoid WAV I/O for batching)
audio, sr = dp.read_wav("file.wav")
clipped = dp.clip(audio, gain_db=8.0)
restored, meta = dp.declip(clipped, sr)
dp.write_wav("out.wav", restored, sr)
```

### If the C++ build fails

The script falls back to pure NumPy automatically. No crash, no error — just slower. Call `build_extension()` with `force=True` to retry after fixing the issue.

---

## Local Development

```bash
cd plugins_port/declipper

# Build C++ extension locally
mkdir -p build && cd build
cmake ../csrc -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp _faurge_declip_cpp*.so ../
cd ..

# Run tests
python3 test/test_clip.py
python3 test/test_declip.py
python3 test/test_quality.py

# CLI
python3 declipper.py declip input.wav output.wav --verbose
python3 declipper.py clip input.wav output.wav --gain-db 8.0
```

### Dependencies for local testing

- Python 3.12, NumPy, SciPy, pybind11
- g++ with C++17, cmake ≥ 3.14

---

## API Reference

### `clip(audio, gain_db=6.0, mode="hard") -> np.ndarray`
Apply gain and clip. Modes: `"hard"` (clamp), `"soft"` (tanh), `"random_segment"`.

### `declip(audio, sample_rate, **config) -> (np.ndarray, dict)`
Declip in-place. Returns restored array + metadata dict.

### `clip_file(input_path, output_path, ...) -> dict`
Read WAV → clip → write WAV.

### `declip_file(input_path, output_path, ...) -> dict`
Read WAV → declip → write WAV.

### `quality_report(original, processed, sr) -> dict`
One-shot: SNR, THD+N before/after, peak, RMS, improvement.

### `build_extension(source_dir, build_dir="/tmp/declipper_build", force=False) -> bool`
Compile C++ .so at runtime.

### `read_wav(path) -> (np.ndarray, int)`
Read WAV to float32 [-1, 1].

### `write_wav(path, audio, sr)`
Write float32 audio to WAV.

---

## Design Decisions

**Why not just use the C++ CLI with subprocess?** Subprocess overhead per file (~50ms) adds up across thousands of files. The pybind11 path passes float arrays directly with zero serialization cost.

**Why not ship a pre-compiled wheel?** Wheels are Python-version-specific and fragile. Compile-at-runtime on Kaggle's known Docker image (Debian 13, Python 3.12, g++ 13) is deterministic and portable.

**Why is the quality check one-shot with no loop?** The original Phase 4 "Fabian Orchestrator" reflection loop was designed for live inference, not training data generation. For training, you just need the one-shot SNR/THD+N measurement to verify the pipeline.

**Why pure NumPy fallback and not just Cython or Numba?** The fallback must work on Kaggle with zero additional setup. NumPy and SciPy are pre-installed on every Kaggle image. Numba requires LLVM compilation; Cython requires a separate build step. NumPy is the baseline that's always available.
