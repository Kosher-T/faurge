# Ursula Training Analysis: Why It Plateaus & What To Do

## The Diagnosis

### What the logs tell us

| Metric | Step 10k | Step 90k | Step 150k |
|--------|----------|----------|-----------|
| Eval Mean MSE | 1909 | 1884 | 1906 |
| Eval Mean Reward | -0.44 | -0.44 | -0.44 |
| % below floor | 0% | 0% | 0% |
| Alpha | 0.47 | 0.0001 | 0.0001 |

The agent converged to a **fixed policy** by step ~90k and has been stuck there ever since. Alpha collapsed to 0.0001 (despite the `ALPHA_MIN = 0.05` floor you added — it seems to not be working properly), meaning the agent stopped exploring entirely. The MSE oscillates between ~1500–2200 per step, which is suspiciously close to the **identity floor region** — the agent likely learned to "do nothing useful" and is outputting near-zero actions.

### The 5 Root Problems

#### 1. **227D Action Space is Too Large for Naive RL**

SAC was designed for action spaces of ~10–30 dimensions. Your 227D space (186 EQ + 14 compressor + 6 esser + 7 saturator + 6 limiter + 6 transient + 2 gain) is enormous. The probability of a random 227D action improving MSE is astronomically low. The agent's only viable strategy during warmup was "do as little as possible."

#### 2. **The Supervised Pretraining (Phase 6A) is Barely Helping**

Your current supervised approach:
- Searches only **500 random candidates** per pair
- Uses only **3 pairs**
- Trains for 200 epochs on 3 data points with noise augmentation

500 random trials in 227D space is like searching a haystack with a thimble. The "best" actions found by random search are almost certainly still terrible — just slightly less terrible than average. The policy learns to predict mediocre actions, which is barely better than doing nothing.

#### 3. **Multi-Step Environment is Counterproductive**

Your environment allows 50 steps per episode, each step applying the plugin chain **sequentially on the output of the previous step**. This means:
- Step 1: Apply EQ+Comp+Sat+... to degraded audio
- Step 2: Apply EQ+Comp+Sat+... **again** to the already-processed audio
- Step 3: ... and again

This is destructive. Audio processing plugins are not idempotent. Applying random compression twice destroys dynamics. Applying random EQ twice amplifies noise. The agent quickly learns that any action makes things worse → outputs near-zero → gets stuck.

#### 4. **Entropy Death Despite Floor**

Your `ALPHA_MIN = 0.05` floor code exists but doesn't seem to be working in the logs — alpha dropped to 0.0001 by step ~10k and stayed there. Looking at the code, the floor is enforced, but with `target_entropy = -113.5` (half of 227), the optimizer drives alpha down before the floor kicks in, and by that point the policy is already committed to a near-deterministic strategy.

#### 5. **You Have the Degradation Params But You're Not Using Them**

This is the key insight. In Phase 2, you save `degradation_params.json` with the **exact parameters used to degrade each audio clip**. You know what was done to the audio. The restoration problem is "undo this degradation," and you have the degradation spec. **You can compute approximate inverse parameters directly** without any random search.

---

## The Strategy: Inverse Degradation Supervised Learning

### Core Idea

You **do** have restoration parameters — they're the **inverse of the degradation parameters**. For most of your plugins, "undoing" a degradation has a clear mathematical relationship:

| Plugin | Degradation | Restoration (Inverse) |
|--------|-------------|----------------------|
| **EQ** | Gain +6 dB at 2kHz | Gain **-6 dB** at 2kHz (same freq, Q, type) |
| **Compressor** | Threshold -30, ratio 4:1 | Threshold ~0 (disabled), ratio ~1:1, or expander |
| **Gain** | +5 dB | **-5 dB** |
| **Saturator** | Drive 12 dB, mix 0.8 | Drive 0, mix 0 (disable) |
| **Limiter** | Ceiling -6 dB | Ceiling 0 dB (disable) |
| **Transient** | Attack +12 dB | Attack **-12 dB** (or disable with mix=0) |
| **De-esser** | Threshold -30 | Threshold 0 (disable) |

### Why This Works

1. **EQ is the dominant effect** — 186 of 227 parameters are EQ. EQ is perfectly invertible: if you boosted +6 dB at 1kHz, cutting -6 dB at 1kHz reverses it exactly.
2. **Non-linear effects (comp, sat, limiter) can't be perfectly inverted**, but "disable them" (drive=0, mix=0, ratio=1:1, ceiling=0) is a strong baseline that removes their damage.
3. **This gives you thousands of high-quality training pairs** instead of 3 pairs with random search targets.

### Implementation Plan

> [!IMPORTANT]
> This replaces Phase 6A entirely and fundamentally changes the RL training setup.

#### Phase 6A-v2: Inverse Degradation Dataset

```
For each pair in degradation_params.json:
    1. Load the degradation params dict
    2. Compute inverse params:
       - EQ: negate all gain_db values, keep freq/Q/type
       - Compressor: set threshold=0, ratio=1.0, wet_dry=0
       - Saturator: set drive=0, mix=0
       - Limiter: set ceiling=0
       - Transient: negate gain values OR set mix=0
       - De-esser: set threshold=0
       - Gain: negate gain_db
    3. Encode inverse params → 227D action via ActionUnnormalizer.encode()
    4. Build observation from the metric pair
    5. Store (observation, inverse_action) as supervised target
```

This gives you potentially **hundreds or thousands** of training pairs with **physically meaningful** targets, not random search noise.

#### Phase 6A-v2: Training Changes

```python
# Instead of random search targets:
SUPERVISED_EPOCHS = 500        # more epochs since we have good targets
SUPERVISED_LR = 3e-4           # lower LR for cleaner convergence
MAX_PAIRS = None               # use ALL pairs
AUGMENTATION_NOISE = 0.01     # smaller noise since targets are meaningful
```

#### Phase 6 (RL): Critical Changes

1. **Single-step episodes** (`MAX_STEPS = 1`): The agent gets one shot to fix the audio. No cascading destruction.
2. **Much larger warmup** (`WARMUP_STEPS = 50_000`): Fill the buffer with diverse experience before learning.
3. **Stronger alpha floor** (`ALPHA_MIN = 0.2`): Force continued exploration.
4. **Reduce the effective action space**: Freeze the 186 EQ dimensions for the first 50k steps and only train the 41 non-EQ parameters. Unfreeze EQ later.
5. **Step-relative reward**: Reward delta-MSE (improvement over previous state) rather than absolute position relative to init.

---

## Concrete Code Changes

### 1. New Phase 6A: `compute_inverse_params()`

The key new function — converts degradation params to restoration action vector:

```python
def compute_inverse_action(deg_params):
    """Convert degradation parameters to an approximate inverse action vector."""
    inverse = np.zeros(OUTPUT_DIM, dtype=np.float32)
    
    # EQ: negate gains, keep frequencies and Q
    for b in range(31):
        prefix = f"eq_band{b+1}"
        # Only negate if the degradation actually used this band
        # (degradation uses 1-6 random bands, rest are identity)
        gain = deg_params['eq_bands'][b]['gain_db'] if b < len(deg_params['eq_bands']) else 0.0
        freq = deg_params['eq_bands'][b]['freq_hz'] if b < len(deg_params['eq_bands']) else 1000.0
        q = deg_params['eq_bands'][b]['q'] if b < len(deg_params['eq_bands']) else 1.0
        
        # Encode as normalized [-1, 1] action
        # Inverse: negate the gain
        inverse_gain = -gain
        # ... encode freq, inverse_gain, q, etc. to [-1,1] using param ranges
    
    # Compressor: disable it (threshold=0, ratio=1, wet_dry=0)
    # ... encode comp_threshold=0, comp_ratio=1, comp_wet_dry=0
    
    # Saturator: disable (drive=0, mix=0)
    # Limiter: disable (ceiling=0)
    # Transient: disable (mix=0) or negate gains
    # Gain: negate
    inverse_gain_db = -deg_params['gain']['gain_db']
    
    return inverse  # 227D vector in [-1, 1]
```

### 2. Single-Step Environment

```python
MAX_EPISODE_STEPS = 1  # ONE shot to fix the audio
```

### 3. Hierarchical Action Space (Phased Training)

```python
# Phase 1 (steps 0-50k): Only train non-EQ params (41D)
# Phase 2 (steps 50k-150k): Unfreeze EQ, full 227D
# This makes the initial learning problem tractable
```

### 4. Fix Alpha Floor

```python
# In SACAgent.update(), after alpha_optimizer.step():
with torch.no_grad():
    min_log_alpha = torch.tensor([np.log(ALPHA_MIN)], device=self.device)
    self.log_alpha.data = torch.maximum(self.log_alpha.data, min_log_alpha)
```

---

## Answer to Your Key Question

> Can we train the model to predict restoration parameters from degradation parameters even though we don't have the restoration parameters?

**Yes — because for your pipeline, the inverse of the degradation parameters IS the restoration parameters.** You synthesized the degradation yourself using known, recorded parameters. The inverse is computable:

- **EQ inverse**: Exact (negate gains)
- **Gain inverse**: Exact (negate dB)
- **Non-linear effects inverse**: Approximate (disable them → removes ~90% of their damage)

The remaining non-linear residual is what RL can learn to clean up **after** the supervised pretraining gives it a strong starting point.

---

## Priority Order

1. ☐ **Build inverse degradation dataset** from `degradation_params.json` — highest impact
2. ☐ **Switch to single-step episodes** — removes cascading destruction
3. ☐ **Fix alpha floor** — restore exploration
4. ☐ **Retrain supervised on full dataset** with inverse targets
5. ☐ **RL fine-tuning** with pretrained weights, strong alpha floor, curriculum

> [!TIP]
> If the inverse-degradation supervised model already gets MSE below the identity floor on eval, you may not even need RL at all — just export the supervised policy directly.
