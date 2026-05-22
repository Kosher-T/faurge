# Ursula's Plugin Suite — Implementation Plan

## Summary

Build seven headless DSP plugins that form Ursula's "hands." Each plugin is dual-use:

- **Degradation**: Ruin pristine audio during training data generation
- **Restoration**: Apply Ursula's output parameters during inference to fix audio

Each follows the pattern established by `plugins/declipper/`, `plugins/denoiser/`, and `plugins/exciter/`: C++17, CMake, libsndfile, standalone CLI executable + library, custom test framework.

---

## Plugin 1: Parametric EQ

Controls spectral shaping — Ursula's primary tool for closing the LTAS gap.

### Parameters (31 bands × 6 per band)

| Per band | Parameter | Range | Description |
|----------|-----------|-------|-------------|
| ×31 | `freq_hz` | 20–20000 | Center frequency |
| ×31 | `gain_db` | -24–+24 | Boost/cut |
| ×31 | `q` | 0.1–10 | Bandwidth |
| ×31 | `filter_type` | {peak, low_shelf, high_shelf, highpass, lowpass, bandpass, notch} | 7-way categorical |
| ×31 | `stereo_skew_db` | -6–+6 | L/R gain difference for this band |
| ×31 | `dynamic_depth` | 0–1 | How much gain varies with input level (0 = static, 1 = fully dynamic EQ) |

Total: 186 continuous + categorical parameters.

### CLI

```
faurge-eq <input.wav> <output.wav> [options]
  --band1-freq 1000 --band1-gain -3.0 --band1-q 1.4 --band1-type peak --band1-skew 0 --band1-dynamic 0
  --band2-freq 200  --band2-gain +2.5 --band2-q 0.7 --band2-type low_shelf ...
  --band31-freq 60  --band31-gain -0.5 --band31-q 1.0 --band31-type peak ...
  --json --verbose
```

### Degradation mode

`faurge-eq-ruin <input.wav> <output.wav>`
Applies 1–6 random bands with random freq/gain/Q/type/skew/dynamic to destroy spectral balance.

---

## Plugin 2: Compressor

Controls dynamic range — Ursula's tool for matching LUFS and crest factor.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `threshold_db` | -60–0 | Level above which compression kicks in |
| `ratio` | 1–20 | Compression ratio (1 = off) |
| `attack_ms` | 0.1–100 | How fast the compressor reacts |
| `release_ms` | 10–1000 | How fast the compressor recovers |
| `knee_db` | 0–12 | Smooth knee width |
| `lookahead_ms` | 0–10 | Lookahead window for anticipatory gain reduction |
| `hold_ms` | 0–200 | Hold time before release phase begins |
| `wet_dry_mix` | 0–1 | Parallel compression blend |
| `stereo_link` | 0–1 | 0 = independent channels, 1 = full link (average detection) |
| `sidechain_hp_hz` | 20–500 | Highpass filter on the detection circuit |
| `sidechain_lp_hz` | 500–20000 | Lowpass filter on the detection circuit |
| `saturate_drive_db` | 0–12 | Extra saturation applied to the compressed signal |
| `output_trim_db` | -12–+12 | Makeup gain after compression |
| `detector_type` | {RMS, peak, feed_forward, feed_back} | 4-way detection mode |

Total: 14 parameters.

### CLI

```
faurge-compress <input.wav> <output.wav> [options]
  --threshold -24 --ratio 4.0 --attack 5 --release 150 --knee 6
  --lookahead 2 --hold 50 --mix 0.8 --link 1.0
  --sidechain-hp 80 --sidechain-lp 18000 --saturate 3.0 --trim +1.5 --detector RMS
  --json --verbose
```

### Degradation mode

`faurge-compress-ruin <input.wav> <output.wav>`
Aggressive compression (high ratio, low threshold, fast attack) or none at all — both degrade.

---

## Plugin 3: Esser (Dynamic Sibilance Processor)

Controls the sibilance band — applies gain reduction/boost only when the monitored band exceeds threshold.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `center_freq_hz` | 4000–10000 | Center of sibilance band |
| `threshold_db` | -60–0 | Level above which processing engages |
| `ratio` | 0.25–20 | < 1 = boost, > 1 = cut, 1 = off |
| `bandwidth_hz` | 500–4000 | Width of monitored band |
| `attack_ms` | 0.1–50 | Reaction speed |
| `release_ms` | 10–500 | Recovery speed |

Total: 6 parameters.

### CLI

```
faurge-esser <input.wav> <output.wav> [options]
  --center 7200 --threshold -30 --ratio 5.0 --bandwidth 2000 --attack 1 --release 100
  --json --verbose
```

### Degradation mode

`faurge-esser-ruin <input.wav> <output.wav>`
Harsh sss boost (ratio < 1) or complete sibilance cut (ratio > 10, low threshold).

---

## Plugin 4: Limiter

Safety ceiling — prevents peaks from exceeding a hard or soft limit.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `ceiling_db` | -12–0 | Maximum allowed peak level |
| `release_ms` | 1–500 | Recovery speed after gain reduction |
| `lookahead_ms` | 0–10 | Lookahead for anticipatory limiting |
| `clip_mode` | {soft, hard} | Soft = knee'd saturation, hard = brickwall |
| `stereo_link` | 0–1 | 0 = independent, 1 = linked |
| `oversampling` | 1–4 | Anti-alias oversampling factor |

Total: 6 parameters.

### CLI

```
faurge-limiter <input.wav> <output.wav> [options]
  --ceiling -1.0 --release 50 --lookahead 2 --mode soft --link 1.0 --oversample 2
  --json --verbose
```

### Degradation mode

`faurge-limiter-ruin <input.wav> <output.wav>`
Extremely low ceiling with hard clip mode.

---

## Plugin 5: Saturator

Harmonic coloration — adds controlled distortion for character.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `drive_db` | 0–24 | Pre-saturation gain |
| `mix` | 0–1 | Wet/dry blend |
| `type` | {tube, tape, diode, asymmetric} | 4-way saturation curve |
| `highpass_freq` | 20–500 | HPF before saturation (prevents muddy bass) |
| `lowpass_freq` | 2000–20000 | LPF after saturation (tames harsh harmonics) |
| `oversampling` | 1–4 | Anti-alias oversampling |
| `output_trim_db` | -12–+12 | Post-saturation level |

Total: 7 parameters.

### CLI

```
faurge-saturate <input.wav> <output.wav> [options]
  --drive 6.0 --mix 0.4 --type tube --hpf 100 --lpf 16000 --oversample 2 --trim -1.0
  --json --verbose
```

### Degradation mode

`faurge-saturate-ruin <input.wav> <output.wav>`
Extreme drive with asymmetric clipping to destroy clarity.

---

## Plugin 6: Transient Shaper

Envelope control — shapes attack and sustain of individual events.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `attack_gain_db` | -24–+24 | Boost or cut the attack transient |
| `sustain_gain_db` | -24–+24 | Boost or cut the sustain tail |
| `attack_time_ms` | 0.1–50 | How long until a transient is considered "attack" |
| `release_time_ms` | 10–500 | How fast to return to unity after a transient |
| `sensitivity` | -30–0 | Minimum transient level to trigger processing |
| `mix` | 0–1 | Wet/dry blend |

Total: 6 parameters.

### CLI

```
faurge-transient <input.wav> <output.wav> [options]
  --attack-gain +3.0 --sustain-gain -2.0 --attack-time 5 --release-time 200 --sensitivity -20 --mix 0.8
  --json --verbose
```

### Degradation mode

`faurge-transient-ruin <input.wav> <output.wav>`
Extreme attack boost + sustain cut to make audio percussive and unnatural, or the reverse to mush it out.

---

## Plugin 7: Gain

Final loudness and balance.

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `gain_db` | -12–+12 | Post-processing gain offset |
| `stereo_balance` | -1–1 | -1 = full left, 0 = center, 1 = full right |

Total: 2 parameters.

### CLI

```
faurge-gain <input.wav> <output.wav> [options]
  --gain +2.5 --balance 0 --json --verbose
```

### Degradation mode

`faurge-gain-ruin <input.wav> <output.wav>`
Extreme gain (+12 dB clipping or -12 dB near-silence), hard-pan or swap channels.

---

## Summary: Ursula's Action Space

| Plugin | Parameters | Total dims |
|--------|-----------|------------|
| Parametric EQ (31 bands) | freq, gain, q, filter_type, stereo_skew, dynamic_depth × 31 | 186 |
| Compressor | threshold, ratio, attack, release, knee, lookahead, hold, wet_dry, stereo_link, sidechain_hp, sidechain_lp, saturate_drive, output_trim, detector_type | 14 |
| Esser | center_freq, threshold, ratio, bandwidth, attack, release | 6 |
| Limiter | ceiling, release, lookahead, clip_mode, stereo_link, oversampling | 6 |
| Saturator | drive, mix, type, highpass_freq, lowpass_freq, oversampling, output_trim | 7 |
| Transient Shaper | attack_gain, sustain_gain, attack_time, release_time, sensitivity, mix | 6 |
| Gain | gain_db, stereo_balance | 2 |
| **Total** | | **227** |

Each parameter is normalized to [-1, 1] and unscaled to its real range by the RL environment wrapper. Categorical parameters (filter_type, detector_type, clip_mode, saturate_type) use softmax over their options.

227D is a large action space. SAC with automatic entropy tuning handles it, but expect extended training time. The curriculum warmup (single-clip → multi-clip) becomes critical — the network must learn the causal map of 227 knobs before it has to generalize across content variation.

---

## Processing Order

```
Input → EQ → Compressor → Esser → Saturator → Limiter → Transient Shaper → Gain → Output
```

This order mirrors typical studio practice: spectral shaping first, then dynamics, then sibilance, then harmonic coloration, then safety ceiling, then envelope finishing, then final level.

---

## Build & File Structure (per plugin)

Each plugin follows the exciter layout exactly:

```
plugins/<name>/
├── CMakeLists.txt
├── include/faurge/
│   ├── <name>.hpp           # Main public API
│   ├── <name>_types.hpp     # Config + Result structs
│   └── <name>_metrics.hpp   # JSON output
├── src/
│   ├── <name>.cpp           # Orchestrator + WAV I/O
│   ├── <name>_metrics.cpp
│   └── main.cpp             # CLI entry point
└── test/
    └── generate_pairs.cpp   # Training pair generator (degradation helper)
```

Dependencies: C++17, libsndfile, no additional libraries.

---

## Dual-Use Pipeline

```
Kaggle (training data gen)
  ┌──────────────────────────────────────┐
  │  pristine.wav                         │
  │    → faurge-eq-ruin (random bands)    │
  │    → faurge-compress-ruin (heavy)     │
  │    → faurge-esser-ruin (harsh sss)    │
  │    → faurge-saturate-ruin (drive)     │
  │    → faurge-limiter-ruin (crush)      │
  │    → faurge-transient-ruin (shape)    │
  │    → faurge-gain-ruin (jitter)        │
  │    → degraded.wav                     │
  └──────────────────────────────────────┘
  Output: {degraded.wav, pristine.wav} pair with metrics

RL Gym Env (training loop)
  ┌──────────────────────────────────────┐
  │  Ursula outputs: [227 params]         │
  │    → faurge-eq      (apply params)    │
  │    → faurge-compress (apply params)   │
  │    → faurge-esser   (apply params)    │
  │    → faurge-saturate (apply params)   │
  │    → faurge-limiter (apply params)    │
  │    → faurge-transient (apply params)  │
  │    → faurge-gain    (apply params)    │
  │    → processed.wav                    │
  │  Reward = -MSE(M_result, M_ref)       │
  └──────────────────────────────────────┘

Inference (live pipe)
  ┌──────────────────────────────────────┐
  │  Ursula (frozen ONNX): [227 params]   │
  │    → same 7 plugins in sequence       │
  │    → A_dsp → Genesis                  │
  └──────────────────────────────────────┘
```


---

## Training Pair Policy

### Same-gender (or same-cluster) pairing

Sources and references must share gender: male → male, female → female. Cross-gender pairs teach voice morphing (e.g., boost lows to make a female voice approximate a male LTAS), not restoration.

**Refinement — LTAS centroid clustering:** Within-gender voice range is still wide (bass vs tenor, alto vs soprano). If artifacts appear, cluster speakers by their LTAS centroid (the average frequency of their spectral energy) and enforce pairs within the same cluster. This naturally groups baritones, tenors, altos, sopranos, etc., tighter than binary gender alone.

### Baseline similarity ceiling

Before training, establish the **identity floor**:

1. Take good recordings of speaker A and speaker B (same gender/cluster).
2. Compute 67D metrics for each.
3. Measure MSE(M(A_i), M(B_j)) across all cross-speaker combinations.
4. Average → identity floor.

This is the irreducible metric distance between two different humans producing good audio. During training, clamp the reward:

```
reward = -soft_clamp(MSE - floor, k)
```

Where `soft_clamp(x, k)` is a smooth sigmoid-like transition (e.g., `k * tanh(x/k)`) rather than a hard `max(x, 0)`. This keeps the gradient smooth near the floor — a hard cutoff at 0 would kill gradient signal abruptly once MSE dips below the floor, preventing the network from learning to improve dimensions that still have room.

The single scalar floor works because MSE distributes gradient per-component: if LTAS MSE is 0.02 (above floor) and LUFS MSE is 0.000 (below floor), the gradient vanishes for LUFS while LTAS continues to be optimized.

---

## Candidate Metrics (Future Expansion)

Potential metrics to add if the initial 67D observation space proves insufficient:

| Metric | Dims | What it captures |
|--------|------|-----------------|
| Spectral centroid | 1 | Perceptual brightness / HF balance |
| Spectral flatness | 1 | Noise-likeness vs tonality |
| Noise floor estimate | 1 | Background hiss level |
| Band energy ratios (3–5 bands) | 3–5 | Coarse spectral balance, more interpretable than full LTAS |

Do not add in the first training pass. Wait until Ursula converges, then evaluate failure cases: if output sounds "off" on a specific perceptual dimension despite good LTAS/LUFS/DR scores, that dimension tells you which metric was missing.
