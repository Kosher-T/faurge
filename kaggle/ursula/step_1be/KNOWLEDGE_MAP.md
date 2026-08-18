# Ursula's Knowledge Map

What each step trained the model to do, what it learned, and what's missing.

---

## Step 0 — 67D Metric Literacy

**Input:** 67D degraded metrics
**Output:** 67D clean metrics
**Degradation:** Various (gain, EQ, compressor)
**Clips:** 5

**What it learned:** How to reconstruct Tier 0 metrics (LTAS64 + LUFS + Crest + ZCR) from degraded audio. The model sees degraded metrics and learns to predict the original clean metrics.

**Result:** 97.6% improvement. Model understands 67D metric space.

**What it didn't learn:** No plugin parameter prediction. No Tier 1 features.

---

## Step 1a — Gain Prediction (1 clip)

**Input:** 134D (67 degraded + 67 clean metrics)
**Output:** 1D (gain_db)
**Degradation:** Gain-only (±12dB)
**Clips:** 1

**What it learned:** Given degraded+clean metric pairs, predict the gain that was applied. First time predicting plugin parameters.

**Result:** MAE 0.25dB, 97.5% within ±1dB.

**Limitation:** Single clip — no generalization to unseen speakers.

---

## Step 1b — Gain Prediction (5 clips)

**Input:** 134D (67 degraded + 67 clean metrics)
**Output:** 1D (gain_db)
**Degradation:** Gain-only (±12dB)
**Clips:** 5 (3m + 2f)

**What it learned:** Same as 1a but across 5 speakers. Model generalizes gain prediction to unseen speakers.

**Result:** MAE 0.06dB, 100% within ±1dB.

**Limitation:** Only predicts global gain. No EQ.

---

## Step 1ba — EQ Gain Causality

**Input:** 134D (67 degraded + 67 clean metrics)
**Output:** 1D (eq_gain_db)
**Degradation:** EQ at specific frequencies (24-frequency curriculum)
**Clips:** 5

**What it learned:** Given degraded+clean metrics, predict the EQ gain that was applied. First time predicting EQ parameters.

**Result:** Passed 24-frequency curriculum.

**Limitation:** Only predicts EQ gain (not frequency). Fixed frequency per training run.

---

## Step 1bb — Frequency-Aware Gain

**Input:** 135D (67 degraded + 67 clean + freq_index)
**Output:** 1D (eq_gain_db)
**Degradation:** EQ at specific frequencies (50-frequency curriculum)
**Clips:** 5

**What it learned:** Predict EQ gain when told which frequency was used. The freq_index is provided as input.

**Result:** 100% seen frequencies, 18.9x worse on unseen frequencies.

**Limitation:** Can't predict frequency — it's given as input. Doesn't generalize to unseen frequencies.

---

## Step 1bc — 80D Metric Literacy (Stream A)

**Input:** 80D degraded metrics (data-dependent normalized)
**Output:** 80D clean metrics
**Degradation:** Gain-only
**Clips:** 10 (5m + 5f)

**What it learned:** Reconstruct 80D metrics (Tier 0: 67D + Tier 1: 13D) from degraded metrics. Data-dependent min-max normalization. Tier 1 features: Centroid, Bandwidth, Flatness, Flux, Rolloff, Skewness, Kurtosis, Slope + Band Energy Ratios (5).

**Result:** Passed. Model learns all 80D metrics.

**What it didn't learn:** Tier 1 features barely changed with gain-only degradation. Model learned to predict mean.

---

## Step 1bd — Frequency + Gain Prediction (Stream A)

**Input:** 160D (80 degraded + 80 clean metrics)
**Output:** 2D (norm_log_freq, norm_gain)
**Degradation:** EQ at specific frequencies (147-frequency curriculum) + no global gain
**Clips:** 10 (5m + 5f)
**Samples:** 735K (10 clips × 147 freqs × 500 degradations)

**What it learned:** Given degraded+clean 80D metrics, predict which EQ frequency and gain were applied. First time predicting BOTH frequency and gain together.

**Result:** Overall Freq MAE 186 Hz (5.3%), Gain MAE 0.49 dB, ±1dB: 92.1%.

**Generalization (frequency-aware split):**
- Seen frequencies: freq=5.3%, gain=0.46 dB
- Unseen frequencies: freq=5.9%, gain=0.48 dB
- Gap: +0.6% freq, +0.02 dB gain — essentially zero

**Key insight:** Model generalizes perfectly to unseen frequencies. The 80D metric representation captures enough information to predict EQ parameters across the full frequency range.

**Limitation:** Single band only. No global gain. Trained on 10 speakers — limited speaker diversity.

---

## Step 1be — RMS Metric Literacy + Multi-Clip EQ Prediction (revised)

**Input:** 162D = [degraded_81D, reference_81D] (81D = 80D + RMS Energy)
**Output:** Phase 1: 81D (reference metrics) | Phase 2: 3D (freq, eq_gain, global_gain)
**Degradation:** Phase 1: EQ only (no global gain) | Phase 2: EQ + global gain
**Clips:** 8 train + 2 test (DAPS + VCTK, dynamic speaker selection)
**Frequencies:** 147 log-spaced (120 train, 27 unseen for generalization testing)
**Samples:** Phase 1: 8 clips × 200 degradations = 1,600 | Phase 2: 8 clips × 120 freqs × 200 = 192,000

**What it tried to do:** Address the three gaps identified in step 1bd:
1. **Tier 1 features not learned** — Phase 1 uses EQ degradation (not gain-only) so spectral shape actually changes
2. **No global gain separation** — RMS energy added to help model distinguish signal level from spectral shape
3. **Speaker diversity** — 8+ speakers from DAPS + VCTK instead of 10 from one dataset

**Phase 1 — Metric Literacy (162D → 81D):**
- Trains encoder to reconstruct clean 81D metrics from degraded+reference pair
- EQ degradation forces model to learn how spectral features change with EQ
- Saves encoder weights for Phase 2 initialization

**Phase 2 — EQ Prediction (162D → 3D):**
- Loads Phase 1 encoder, adds prediction head
- Predicts [freq, eq_gain, global_gain] from degraded+reference pair
- Encoder unfreezes during training
- Loss weights: freq=0.35, eq_gain=0.35, global_gain=0.30

**Eval:** 2×2 matrix (seen/unseen clips × seen/unseen frequencies)

**Status:** Not yet run on Kaggle. Previous attempt (original step_1be with 3 clips, 2 EQ bands) failed — model learned global gain only.

---

## What Ursula Knows (Summary)

| Capability | Status | Step |
|------------|--------|------|
| Tier 0 metric reconstruction (67D) | ✅ Learned | Step 0 |
| Tier 1 metric reconstruction (13D) | ⚠️ Partially learned (gain-only doesn't change them) | Step 1bc |
| Global gain prediction | ✅ Learned (MAE 0.06dB) | Step 1b |
| EQ gain prediction (single band) | ✅ Learned | Step 1ba |
| EQ frequency prediction | ✅ Learned (5.3% MAE) | Step 1bd |
| EQ freq + gain (joint) | ✅ Learned, generalizes to unseen freqs | Step 1bd |
| RMS energy | 🔜 Planned (81D) | Step 1be |
| Multi-clip EQ + global gain | 🔜 Planned (3D) | Step 1be |
| Unseen speaker generalization | 🔜 Planned (8+ speakers) | Step 1be |

---

## The Gap

Step 1bd proved the model CAN predict EQ frequency + gain from 80D metrics, and generalizes to unseen frequencies. But:

1. **Tier 1 features weren't truly learned** — gain-only degradation doesn't change spectral shape, so the model predicts the mean. The encoder doesn't understand Tier 1.

2. **No global gain separation** — the model hasn't learned to separate "how loud is the signal" (global gain) from "what's the spectral shape" (EQ). RMS energy was added to help with this.

3. **Speaker diversity matters** — 2 speakers failed (original step_1be). 8+ speakers needed for unseen-clip generalization.

---

## What Step 1be Should Teach

Step 1be addresses all three gaps:

1. **Teach Tier 1 features for real** — Phase 1 uses EQ degradation (not gain-only), forcing the model to learn how spectral features change when EQ is applied.

2. **Separate global gain from EQ** — RMS energy (81D) helps the model distinguish signal level from spectral shape. Phase 2 trains on EQ + global gain jointly.

3. **Generalize to unseen speakers** — 8 train + 2 test speakers from DAPS + VCTK. Dynamic speaker selection (no fixed gender split).

**Key design choice:** Both phases use 162D input = [degraded_81D, reference_81D]. The reference metrics give the model a target to compare against — it learns "here's what the audio should sound like, here's what it sounds like now, predict what happened to it."
