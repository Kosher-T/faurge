# Faurge Specialist: De-Clipper

> **Algorithmic Cubic/Spline Waveform Reconstruction** — Standard C++ DSP

Part of the Faurge **Phase 1 Blind Triage** tier. Reconstructs chopped
waveforms when a user overloads their microphone input, restoring the
missing peaks that were destroyed by the digital ceiling.

## Architecture

The de-clipper is a 5-stage pipeline:

```
Raw Audio → Detection → Boundary Analysis → Reconstruction → Post-Filter → Clean Audio
                                                                  ↓
                                                             Metrics/JSON
```

### Stage 1: Detection
- **Hard-clip**: consecutive samples at ±threshold (default 0.9999)
- **Soft-clip**: second-derivative discontinuity detection for analog limiters
- Region merging (clips separated by ≤3 samples are combined)

### Stage 2: Boundary Analysis
- Extracts N anchor samples on each side of the clip
- Computes entry/exit slopes via finite difference
- Estimates missing peak amplitude via parabolic fit

### Stage 3: Reconstruction (3-Tier)
| Clip Length    | Strategy              | Method                              |
|---------------|-----------------------|-------------------------------------|
| ≤16 samples   | Cubic Hermite Spline  | C1-continuous, 4 anchors per side   |
| 17–64 samples | Akima Sub-Spline      | Piecewise cubic, no Runge oscillation |
| >64 samples   | Burg AR Extrapolation | Forward-backward prediction blend   |

### Stage 4: Post-Processing
- 8-sample raised-cosine crossfade at clip boundaries
- 2nd-order Butterworth anti-alias on reconstructed regions
- 1st-order DC blocker at 10 Hz

### Stage 5: Metrics & Reporting
- Clip count, severity classification, per-region overshoot estimate
- Before/after THD+N measurement
- JSON output for training pipeline integration

## Building

### Prerequisites
```bash
sudo apt install libsndfile1-dev cmake g++
```

### Build
```bash
cd specialists/declipper
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests
```bash
cd build
ctest --output-on-failure
```

## Usage

### De-Clip a WAV File
```bash
./faurge-declip clipped.wav clean.wav --verbose
```

### Generate Training Pairs (for cloud training)
```bash
# Create a hard-clipped version of clean audio (+6 dB gain)
./faurge-generate-clipped clean_voice.wav clipped_voice.wav --clip-db 6.0

# Create with random segment clipping
./faurge-generate-clipped clean.wav clipped.wav --random-segments --segment-chance 0.3

# Create soft-clipped version (tanh saturation)
./faurge-generate-clipped clean.wav soft_clipped.wav --soft-clip --clip-db 3.0
```

### JSON Output (for pipeline integration)
```bash
./faurge-declip input.wav output.wav --json > metrics.json
```

### Full Options
```
faurge-declip <input.wav> <output.wav> [options]
  --threshold <float>   Clip detection threshold    (default: 0.9999)
  --merge-gap <int>     Max merge gap in samples    (default: 3)
  --anchor-size <int>   Boundary context samples    (default: 4)
  --crossfade <int>     Crossfade width in samples  (default: 8)
  --overshoot <float>   Peak overshoot multiplier   (default: 1.15)
  --ar-order <int>      AR model order              (default: 14)
  --no-soft-clip        Disable soft-clip detection
  --no-anti-alias       Disable anti-alias filter
  --json                Output metrics as JSON
  --verbose             Detailed per-region logging
```

## Integration with Faurge

This specialist is triggered by **Fabian** during Pre-Flight Triage:

```cpp
if (Peak_Current >= -0.1 /* dBFS */) { use_declipper = true; }
```

In the live pipeline (Phase 4), Fabian calls the de-clipper on the raw
PipeWire buffer before passing the Cleaned Audio to Ursula and Genesis.

For cloud training (Phase 2–3), use `faurge-generate-clipped` to create
paired training data so the Dyad learns to work with de-clipped signals.

## API (C++ Library)

```cpp
#include <faurge/declipper.hpp>

// In-memory processing
faurge::DeclipConfig config;
config.verbose = true;
faurge::Declipper dc(config);

std::vector<float> audio = /* your samples */;
auto result = dc.process(audio, 48000);  // modifies audio in-place

// File-based processing
auto result = dc.processFile("input.wav", "output.wav");
```

## File Structure

```
specialists/declipper/
├── CMakeLists.txt
├── README.md
├── include/faurge/
│   ├── clip_detector.hpp     # Stage 1: Detection
│   ├── clip_region.hpp       # Core data structures
│   ├── declipper.hpp         # Main public API
│   ├── metrics.hpp           # Stage 5: Analysis
│   ├── post_filter.hpp       # Stage 4: Post-processing
│   └── reconstructor.hpp     # Stage 3: Spline math
├── src/
│   ├── clip_detector.cpp
│   ├── declipper.cpp         # Orchestrator + WAV I/O
│   ├── main.cpp              # CLI entry point
│   ├── metrics.cpp
│   ├── post_filter.cpp
│   └── reconstructor.cpp
└── test/
    ├── generate_clipped.cpp  # Training pair generator
    ├── test_detector.cpp
    ├── test_pipeline.cpp
    └── test_reconstructor.cpp
```
