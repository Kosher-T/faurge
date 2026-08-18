# Portable Exciter — Faurge

A self-contained, dual-band harmonic exciter plugin designed to run inside **Kaggle Utility Scripts** for large-scale training data generation. Single file. Pure NumPy/SciPy. No build step. Upload it and go.

---

## Why This Exists

### The Problem

We are training two neural networks — **Ursula** and **Genesis** — to restore degraded audio in real time. These models need hundreds of thousands of clean→degraded audio pairs for training. The degradation pipeline must:

1. **Excite** pristine audio to create realistic "before" states (harmonic saturation, sub-octave synthesis)
2. **Process** the same audio to compute training targets (what the models should learn to do)
3. Run at **Python speed** because thousands of files means hours of audio
4. Execute on **Kaggle**, where datasets live and GPU quota is allocated

### The Constraints

| Constraint | Implication |
|---|---|
| **Kaggle Utility Scripts are single `.py` files** | You cannot `pip install` a package, upload a wheel, or rely on multi-file modules. Everything must live in one file that gets copied to `/kaggle/lib/user/`. |
| **NumPy and SciPy are pre-installed** on Kaggle's Debian 13 Docker | The entire DSP pipeline runs in pure NumPy/SciPy. No C++ compilation needed. No fallback — this IS the implementation. |
| **The exciter is pure DSP** | Unlike the denoiser (which needs a neural network) or the declipper (C++ accelerated), the exciter is a signal-processing chain: crossover filters, tanh waveshaping, and biquad LPFs. All map trivially to NumPy/SciPy. |
| **25-second clips at most (typically 7s)** | At 48 kHz that's 1.2M samples max. The 2x oversampling in the high band doubles this, but still easily fits in memory. |

### The Use Case

```python
import exciter as ex                                    # one .py file

for clean_path in thousands_of_wavs:
    result = ex.process_file(f"clean/{clean_path.name}",
                              f"degraded/{clean_path.name}",
                              high_drive_db=6.0,
                              low_drive_db=3.0)
    # → Ursula trains on (clean → degraded) pairs
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Kaggle Notebook                      │
│  import exciter as ex                                    │
│  ex.process_file("in.wav", "out.wav", high_drive_db=6)   │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  exciter.py (single file)                  │
│  ┌─────────────────────────────────────────────────┐     │
│  │  WAV I/O           scipy.io.wavfile             │     │
│  │  Crossover         LR4 (2× Butterworth)         │     │
│  │  High band         tanh + 2x oversampling + AA  │     │
│  │  Low band          rectification + 120 Hz LPF   │     │
│  │  Mix               wet/dry blend + master vol   │     │
│  │  CLI               argparse subcommands          │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Pure NumPy, Everywhere

When you call `process(audio, sr)`, the script runs the full 5-stage pipeline entirely in NumPy/SciPy. There is no C++ extension, no model download, no build step — it works on every Kaggle image out of the box.

---

## The Exciter Algorithm (4 Stages)

The core is ported directly from `plugins/exciter/` (C++17, DSP-only). The algorithm adds harmonic content via two independent bands:

| Stage | What | Method |
|---|---|---|
| 1 | **Crossover Split** | Two independent Linkwitz-Riley 4th-order crossovers (2× cascaded 2nd-order Butterworth). High band defaults to 2 kHz, low band to 200 Hz. Both configurable. Edge cases: ≤20 Hz → all low, ≥Nyquist → all high. |
| 2 | **High-Band Saturator** | 2x zero-order-hold oversampling → gain (dB→linear) → tanh waveshaping → Butterworth 2nd-order anti-alias LPF at 0.45×Nyquist → 2x downsampling. Generates odd-order harmonics via symmetric tanh transfer function. |
| 3 | **Low-Band Sub-Octave Synth** | Full-wave rectification → gain (dB→linear) → Butterworth 4th-order LPF at 120 Hz (2× cascaded 2nd-order) → sub-level scaling. Creates octave-doubled sub-bass content below the original fundamental. |
| 4 | **Mix + Volume** | `output = input + highMix·(highOut - highBuf) + lowMix·(lowOut - lowBuf)`. Wet/dry blending via delta mixing (not parallel bus). Master volume applied last. Clamped to [-1, 1]. |

Metrics (peak, RMS, band energy) are computed before and after for quality assessment.

---

## Files

```
plugins_port/exciter/
├── exciter.py              # ← THE Utility Script. Single file. Copy-paste to Kaggle.
├── README.md               # This file
└── test/
    ├── test_process.py     # 9 tests: passthrough, harmonics, sub-octave, edge cases
    └── test_quality.py     # 5 tests: perfect score, field completeness, stereo
```

No `csrc/` directory — the exciter is pure DSP implemented entirely in NumPy/SciPy. There is no C++ build step and no model download.

---

## Kaggle Workflow

### First-time setup (once per notebook session)

```python
import exciter as ex

# No build step. No model download. Import and go.
```

### Daily use

```python
# Process a single file
result = ex.process_file("input.wav", "output.wav",
                          high_drive_db=6.0, high_mix=0.5)
print(result["processing_time_ms"], "ms")

# Quality assessment (one-shot, no loop)
report = ex.quality_report(original, processed, sr)
# → {"snr_db": 18.4, "peak_before": 0.95, "peak_after": 0.85, ...}

# In-memory (avoid WAV I/O for batching)
audio, sr = ex.read_wav("file.wav")
processed, meta = ex.process(audio, sr, high_drive_db=6.0)
ex.write_wav("out.wav", processed, sr)

# Disable high or low band
result = ex.process_file("in.wav", "out.wav", high_enable=False)

# Control sub-octave injection
result = ex.process_file("in.wav", "out.wav",
                          low_drive_db=6.0, low_sub_level=0.8)
```

### Batch processing

```python
import glob
import exciter as ex

paths = glob.glob("/kaggle/input/dataset/clean/*.wav")

for p in paths:
    result = ex.process_file(p, f"/kaggle/working/degraded/{p.name}",
                              high_drive_db=6.0)
```

---

## Local Development

```bash
cd plugins_port/exciter

# Dependencies
pip install numpy scipy

# Run tests
python3 test/test_process.py
python3 test/test_quality.py

# CLI
python3 exciter.py excite input.wav output.wav --high-drive 6
python3 exciter.py excite input.wav output.wav --high-drive 6 --json
python3 exciter.py quality original.wav processed.wav
```

### Dependencies for local testing

- Python 3.10+, NumPy, SciPy

---

## API Reference

### `process(audio, sample_rate, **config) -> (np.ndarray, dict)`
Process audio through the dual-band exciter. Config parameters:

| Param | Default | Description |
|---|---|---|
| `high_drive_db` | 3.0 | High-band pre-saturation gain |
| `high_mix` | 0.50 | High-band wet/dry mix [0, 1] |
| `high_crossover_hz` | 2000.0 | High-band crossover frequency |
| `high_enable` | True | Enable high-band |
| `low_drive_db` | 0.0 | Low-band pre-rectification gain |
| `low_mix` | 0.35 | Low-band wet/dry mix [0, 1] |
| `low_crossover_hz` | 200.0 | Low-band crossover frequency |
| `low_sub_level` | 0.50 | Sub-octave injection level |
| `low_enable` | True | Enable low-band |
| `master_volume` | 1.0 | Master volume multiplier |

Returns processed array + metadata dict with peak/RMS/band energy.

### `process_file(input_path, output_path, **config) -> dict`
Read WAV → process → write WAV. Aggregates band energies across channels.

### `quality_report(original, processed, sr) -> dict`
One-shot: SNR, peak, RMS, improvement. Multi-channel averaged to mono.

### `read_wav(path) -> (np.ndarray, int)`
Read WAV to float32 [-1, 1].

### `write_wav(path, audio, sr)`
Write float32 audio to WAV.

---

## Design Decisions

**Why no C++ extension?** Unlike the declipper (which benefits from C++-accelerated Burg AR extrapolation), the exciter's DSP chain — crossover filters, tanh, rectification, biquad LPFs — maps cleanly to vectorized NumPy/SciPy. A C++ extension would save ~2-3ms per file, not worth the build complexity.

**Why two independent crossovers instead of one?** The high and low bands have different crossover frequencies (default 2 kHz vs 200 Hz). A single crossover at one frequency can't serve both. Two independent crossovers allow separate control of the frequency range each band operates on.

**Why delta mixing instead of parallel bus?** `output = input + highMix·(highOut - highBuf)` means when mix=0, the output is identical to input regardless of what the saturation does. Parallel bus mixing would require dry/wet volume balancing and doesn't guarantee passthrough at mix=0.

**Why is the quality check one-shot with no loop?** Same rationale as the declipper and DFN3 — for training data generation, a single SNR measurement is sufficient. The original "Fabian Orchestrator" reflection loop is for live inference, not dataset creation.
