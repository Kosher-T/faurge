# Faurge Plugin: Denoiser

> **DeepFilterNet 3 Noise & Reverb Suppression** — Rust DNN + C++ Wrapper

Part of the Faurge **Phase 1 Blind Triage** tier. Suppresses background noise and reverb in real-time using the DeepFilterNet 3 neural network via a C-ABI Rust bridge.

## Architecture

The denoiser is a 3-stage pipeline:

```
Raw Audio → Noise Estimation → DeepFilterNet 3 → Metrics → Clean Audio
                                     ↓
                               Resampling (→ 48kHz)
                                     ↓
                              Frame-based STFT/ISTFT
```

### Stage 1: Noise Floor Estimation
- Martin's minimum statistics algorithm
- RMS energy in overlapping 1024-sample windows
- Pre/post SNR estimation for metrics reporting

### Stage 2: DeepFilterNet 3 (via Rust Bridge)
- Real-time frame-based processing via `df_bridge` C-ABI crate
- Internal ring buffer for arbitrary-size input handling
- Model runs at 48kHz native, handles non-48kHz via linear resampling
- Configurable attenuation limit (0.0 = bypass, 1.0 = max suppression)

### Stage 3: Metrics & Reporting
- Input/output SNR estimation
- Noise floor measurement (dBFS)
- Noise Reduction Ratio (NRR)
- Processing time tracking
- JSON output for pipeline integration

## Building

### Prerequisites
```bash
sudo apt install libsndfile1-dev cmake g++ cargo rustc
```

Install Rust toolchain if not present:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build
```bash
cd plugins/denoiser
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The build will:
1. Download DeepFilterNet 3 model weights (~3.5MB)
2. Build the Rust bridge crate (`df_bridge`) via `cargo build --release`
3. Compile the C++ sources and link against `libdf_bridge.a`

### Run Tests
```bash
cd build
ctest --output-on-failure
```

## Usage

### Denoise a WAV File
```bash
./faurge-denoise noisy.wav clean.wav --verbose
```

### JSON Output (for pipeline integration)
```bash
./faurge-denoise input.wav output.wav --json > metrics.json
```

### Generate Training Pairs
```bash
# Create white noise + clean speech at 10 dB SNR
./faurge-generate-noisy clean.wav noisy.wav --noise-type white --snr-db 10

# Create pink noise + 5 dB SNR + reverb
./faurge-generate-noisy clean.wav noisy.wav --noise-type pink --snr-db 5 --reverb
```

### Full Options
```
faurge-denoise <input.wav> <output.wav> [options]
  --atten-limit <float>   Suppression strength 0.0–1.0  (default: 0.78)
  --model-path <path>     Path to DeepFilterNet model   (default: bundled)
  --json                  Output metrics as JSON
  --verbose               Detailed processing log
```

## API (C++ Library)

```cpp
#include <faurge/denoiser.hpp>

faurge::DenoiseConfig config;
config.attenLimit = 0.85f;
config.verbose = true;
faurge::Denoiser denoiser(config);

std::vector<float> audio = /* your samples */;
auto result = denoiser.process(audio, 48000);  // modifies audio in-place

// File-based processing
auto result = denoiser.processFile("input.wav", "output.wav");
```

## File Structure

```
plugins/denoiser/
├── CMakeLists.txt
├── README.md
├── models/
│   └── DeepFilterNet3.tar.gz          # Downloaded at build time
├── rust/
│   ├── Cargo.toml
│   ├── cbindgen.toml
│   └── src/
│       └── lib.rs                    # C-ABI bridge to DeepFilterNet 3
├── include/faurge/
│   ├── denoise_types.hpp             # Config + Result structs
│   ├── denoiser.hpp                  # Main public API
│   ├── noise_estimator.hpp           # Pure-DSP noise floor estimation
│   └── denoise_metrics.hpp           # JSON + summary reporting
├── src/
│   ├── denoiser.cpp                  # Orchestrator + WAV I/O
│   ├── noise_estimator.cpp           # Martin's minimum statistics
│   ├── denoise_metrics.cpp           # Metrics computation
│   └── main.cpp                      # CLI entry point
└── test/
    ├── generate_noisy.cpp            # Training pair generator
    ├── test_noise_estimator.cpp      # Noise floor estimation tests
    ├── test_bridge.cpp               # Rust FFI lifecycle tests
    ├── test_pipeline.cpp             # Full end-to-end tests
    ├── test_resampling.cpp           # Sample rate conversion tests
    └── test_attenuation.cpp          # Suppression strength tests
```
