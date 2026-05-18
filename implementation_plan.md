# Faurge Plugin: Exciter — Implementation Plan

## Summary

Build `plugins/exciter/` — a dual-band DSP harmonic synthesis plugin for bandwidth extension, as specified in Phase 2 of `docs/phases.md`. The exciter consists of two parallel processing engines: **High-end saturation** (waveshaping harmonic enhancement) and **Low-end sub-octave tracking** (bass extension via frequency halving). Architecture, build system, test framework, and code style are borrowed from the existing `plugins/declipper/` and `plugins/denoiser/` plugins.

## Technical Considerations & Constraints

> **Shared boilerplate** — The CMake build (C++17, libsndfile via pkg-config), the custom test framework (`TEST`/`ASSERT_TRUE` macros with static registration), the WAV I/O in `processFile()`, and the CLI arg parser are all copied verbatim from declipper/denoiser. No new dependencies are introduced.

> **All-DSP, no ML** — Unlike the denoiser (which depends on a Rust DNN bridge), the exciter is pure algorithmic DSP. This means no external model downloads, no Rust compilation, and zero runtime dependencies beyond libsndfile. Build and test times are minimal.

> **Dual-band crossover** — The high and low bands operate in parallel, each with its own crossover frequency. The raw input is split via complementary Linkwitz-Riley 4th-order (LR4) filters at configurable crossover points, processed independently by each engine, then summed back together. This avoids phase cancellation at the crossover seam.

## Open Questions

1. **Oversampling strategy for high-band saturation** — Waveshaping generates harmonics that can alias above Nyquist. The declipper uses 4x oversampling for ISP detection. Should the high-band saturator use simple 2x linear oversampling, or a more CPU-intensive 4x with polyphase anti-alias? **Default to 2x with a 2nd-order Butterworth post-filter (matching declipper's antiAliasFilter). Simpler = better for real-time.**

2. **Sub-octave tracking method** — Two options: (a) full-wave rectification + PLL-style tracking (more robust), or (b) zero-crossing period detection (cheaper but fragile on noisy signals). **Default to full-wave rectification + 4th-order low-pass at 120 Hz to extract the sub-harmonic envelope. Simple, deterministic, and glitch-free.**

## Technical Design

### Architecture / Logic Flow

```
                ┌─────────────────────────────────────┐
Input ────────► │  Crossover LR4 HP (highCrossoverHz) │──► High-Band Saturator ──┐
                │                                     │     (tanh waveshaping,   │
                │  Crossover LR4 LP (highCrossoverHz) │      2x oversample)      │
                └─────────────────────────────────────┘                          │
                                                                                ├─► ► Output
                ┌─────────────────────────────────────┐                          │
                │  Crossover LR4 HP (lowCrossoverHz)  │                          │
                │                                     │──► Low-Band Sub-Octave  ──┘
                │  Crossover LR4 LP (lowCrossoverHz)  │     (full-wave rect. +
                └─────────────────────────────────────┘      120 Hz LPF + mix)
```

### Interface Definitions

```cpp
// include/faurge/exciter_types.hpp

namespace faurge {

struct ExciterConfig {
    // High band — harmonic saturation
    float highDriveDb     = 3.0f;    // Pre-saturation gain (dB)
    float highMix         = 0.50f;   // Wet/dry mix, 0–1 range
    float highCrossoverHz = 2000.0f; // High-band crossover frequency
    bool  highEnable      = true;

    // Low band — sub-octave synthesis
    float lowDriveDb      = 0.0f;    // Pre-rectification gain (dB)
    float lowMix          = 0.35f;   // Wet/dry mix, 0–1 range
    float lowCrossoverHz  = 200.0f;  // Low-band crossover frequency
    float lowSubLevel     = 0.50f;   // Sub-octave injection level
    bool  lowEnable       = true;

    // Master
    float masterVolume    = 1.0f;

    bool  jsonOutput      = false;
    bool  verbose         = false;
};

struct ExciterResult {
    float processingTimeMs   = 0.0f;
    float inputPeakDb        = -120.0f;
    float outputPeakDb       = -120.0f;
    float inputRmsDb         = -120.0f;
    float outputRmsDb        = -120.0f;
    float highBandEnergyDb   = -120.0f;
    float lowBandEnergyDb    = -120.0f;
    size_t framesProcessed   = 0;
    bool   success           = false;
    std::string errorMessage;
};

}
```

---

## Implementation Details

### DSP Core

#### [NEW] `plugins/exciter/include/faurge/high_band.hpp`
#### [NEW] `plugins/exciter/src/high_band.cpp`
- **HighBand** class with `void process(const float* input, float* output, size_t numSamples, int sampleRate)`
- Internal oversample buffer (2x), butterworth anti-alias post-filter
- Waveshaping via `tanh(drive * sample)` for even-order saturation
- Optional asymmetric transfer function for 2nd-harmonic flavour
- Handles drive=0 (bypass) edge case: identity passthrough
- Handles empty buffer edge case: no-op

#### [NEW] `plugins/exciter/include/faurge/low_band.hpp`
#### [NEW] `plugins/exciter/src/low_band.cpp`
- **LowBand** class with `void process(const float* input, float* output, size_t numSamples, int sampleRate)`
- Sub-octave extraction chain:
  1. Full-wave rectification: `abs(sample)`
  2. 4th-order Butterworth LPF at 120 Hz to isolate envelope
  3. Scale by `subLevel` and `drive`
- Edge case: silence in → silence out (no runaway oscillation)
- Edge case: very low sample rate (< 4000 Hz) → clamp crossover to avoid degenerate filters

### Orchestrator

#### [NEW] `plugins/exciter/include/faurge/exciter.hpp`
#### [NEW] `plugins/exciter/src/exciter.cpp`
- **Exciter** class, mirrors `Declipper`/`Denoiser` API exactly:
  - `ExciterResult process(std::vector<float>& audio, int sampleRate)`
  - `ExciterResult processFile(const std::string& inputPath, const std::string& outputPath)`
- Implements LR4 crossover splitting:
  - Two cascaded 2nd-order Butterworth filters per band (high-pass + low-pass)
  - Process high band → `HighBand`, low band → `LowBand`
  - Sum processed bands
- Edge case: both bands disabled → pure passthrough
- Edge case: empty audio → return `success=false` with error message

### Crossover Filter

#### [NEW] `plugins/exciter/include/faurge/crossover_filter.hpp`
#### [NEW] `plugins/exciter/src/crossover_filter.cpp`
- **CrossoverFilter** utility class, not exposed in public API
- 4th-order Linkwitz-Riley (two cascaded 2nd-order Butterworth stages)
- `void process(const float* input, float* lowOut, float* highOut, size_t numSamples, int sampleRate, float crossoverHz)`
- Edge case: `crossoverHz >= sampleRate/2` → all content goes to low band
- Edge case: `crossoverHz <= 20` → all content goes to high band

### Metrics / Reporting

#### [NEW] `plugins/exciter/include/faurge/exciter_metrics.hpp`
#### [NEW] `plugins/exciter/src/exciter_metrics.cpp`
- Mirrors `DenoiseMetrics` / `Metrics` patterns:
  - `static std::string toJson(const ExciterResult& result)`
  - `static void printSummary(const ExciterResult& result)`

### CLI

#### [NEW] `plugins/exciter/src/main.cpp`
- Mirrors `declipper/src/main.cpp` and `denoiser/src/main.cpp`:
  - `faurge-excite <input.wav> <output.wav> [options]`
  - CLI args: `--high-drive`, `--high-mix`, `--high-cross`, `--low-drive`, `--low-mix`, `--low-cross`, `--low-sub`, `--no-high`, `--no-low`, `--json`, `--verbose`, `--help`

### Training Pair Generator

#### [NEW] `plugins/exciter/test/generate_pairs.cpp`
- Mirrors `generate_clipped.cpp` / `generate_noisy.cpp`
- `faurge-generate-excite-pairs <clean.wav> <excited.wav> [options]`
- Applies controlled harmonic distortion and sub-octave to create pairs for cloud training

### Build System

#### [NEW] `plugins/exciter/CMakeLists.txt`
- Minimal: C++17, PkgConfig, libsndfile, no extra deps
- Targets: `faurge-excite` (main), `faurge-generate-excite-pairs` (training), individual test executables

---

## Environment & Build

- **No new dependencies.** libsndfile is already required by both existing plugins.
- Build procedure identical to declipper:
  ```bash
  cd plugins/exciter && mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  ctest --output-on-failure
  ```

---

## Test & Verification Plan

#### [NEW] `plugins/exciter/test/test_high_band.cpp`
- `test_high_band_bypass_at_zero_drive`: With drive=0, output should match input within floating-point tolerance.
- `test_high_band_generates_harmonics`: Sine tone at 500 Hz through saturator should produce measurable 1000 Hz (2nd harmonic) component.
- `test_high_band_does_not_clip`: output samples stay within [-1, 1] even with extreme drive values.
- `test_high_band_no_self_oscillation_on_silence`: Silence in → a few floating-point residuals but no growing oscillation.
- `test_high_band_drive_increases_energy`: RMS energy monotonically increases with drive parameter.

#### [NEW] `plugins/exciter/test/test_low_band.cpp`
- `test_low_band_generates_sub_octave`: A 200 Hz sine produces a measurable 100 Hz component at the output.
- `test_low_band_bypass_at_zero_mix`: With mix=0, output should match input.
- `test_low_band_silence_in_silence_out`: All-zero input produces all-zero output.
- `test_low_band_sub_level_controls_energy`: Output low-frequency energy scales with subLevel parameter.

#### [NEW] `plugins/exciter/test/test_pipeline.cpp`
- `test_full_pipeline_increases_high_freq_energy`: Processing a broadband signal increases spectral energy in the high band.
- `test_full_pipeline_increases_low_freq_energy`: Processing a bass signal increases spectral energy in the low band.
- `test_pipeline_passthrough_at_minimal_settings`: With all drives=0, mix=0, output closely matches input.
- `test_pipeline_no_clipping`: Output samples stay within [-1, 1] for reasonable input.
- `test_processing_time_is_reasonable`: 1 second at 48kHz processes in under 500 ms.

#### [NEW] `plugins/exciter/test/test_metrics.cpp`
- `test_json_output_is_valid`: JSON string contains all expected fields.
- `test_metrics_are_reasonable`: Peak/RMS fields are in valid ranges, processing time is non-negative.

### Manual Verification
1. `./faurge-excite input_sine.wav output.wav --verbose` — verify console output shows band energies.
2. `./faurge-excite input.wav output.wav --json | python -m json.tool` — validate JSON structure.
3. `./faurge-generate-excite-pairs clean.wav excited.wav --high-drive 6` — verify output file was written correctly with libsndfile.
4. Listen-test: a 100 Hz sine should produce audible sub-bass; a 5 kHz sine should produce brighter harmonics.
5. Spectrum analysis with `ffmpeg` or `sox`: confirm harmonic peaks at 2×, 3× the input frequency in the high band.

---

## File Structure

```
plugins/exciter/
├── CMakeLists.txt
├── README.md
├── include/faurge/
│   ├── crossover_filter.hpp     # LR4 crossover utility
│   ├── exciter.hpp              # Main public API
│   ├── exciter_types.hpp        # Config + Result structs
│   ├── exciter_metrics.hpp      # JSON + summary reporting
│   ├── high_band.hpp            # High-frequency saturator
│   └── low_band.hpp             # Low-frequency sub-octave synth
├── src/
│   ├── crossover_filter.cpp
│   ├── exciter.cpp              # Orchestrator + WAV I/O
│   ├── exciter_metrics.cpp
│   ├── high_band.cpp
│   ├── low_band.cpp
│   └── main.cpp                 # CLI entry point
└── test/
    ├── generate_pairs.cpp       # Training pair generator
    ├── test_high_band.cpp       # High-band unit tests
    ├── test_low_band.cpp        # Low-band unit tests
    ├── test_pipeline.cpp        # Full integration tests
    └── test_metrics.cpp         # Metrics correctness tests
```
