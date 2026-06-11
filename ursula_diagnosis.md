# Ursula SL Training Diagnosis

> [!CAUTION]
> **ROOT CAUSE IDENTIFIED: ~90% of the 125D target vector is CONSTANT across all samples.**
> The model quickly memorizes these constants, then plateaus because the remaining ~10% of useful signal is drowned out.

## The Smoking Gun: Why Action Norm ≈ 7.2

The degradation pipeline generates **1–6 random EQ bands** out of 31 total slots. The inverse-action target fills **unused bands with fixed identity values**:

| Parameter | Real Value | Encoded Raw Value |
|-----------|-----------|-------------------|
| `freq`    | 1000 Hz   | ≈ 0.424 (log-encoded) |
| `gain`    | 0 dB      | 0.0 |
| `q`       | 1.0       | ≈ −0.818 |
| `filter_type` | peak (0) | −1.0 |

Each identity band contributes a norm of **≈ 1.38** to the action vector. With ~25–28 identity bands per sample:

```
28 bands × 4 params/band = 112 constant dimensions
√(28 × 1.38²) ≈ 7.3 — matches the observed 7.2 action norm perfectly
```

The action norm is **not from learning** — it's a mathematical consequence of encoding identity EQ into the target.

## Why Training Plateaus at Test MSE ≈ 0.056

### Phase 1 (epochs 1–10): Fast drop
The model learns the **constant background** (identity bands = 112 of 125 dims). This is trivial — just memorize one fixed vector.

### Phase 2 (epochs 10+): Permanent plateau
Once the constants are learned, the remaining loss comes from:
- **~12 active EQ dims** (3 bands × 4 params) — varies per sample  
- **1 gain dim** — varies per sample

These 13 informative dims are only **10.4%** of the output vector. The MSE contribution from the variable dims is overwhelmed by the constant dims in the gradient signal.

```
MSE = (112 constant dims × ~0 error + 13 variable dims × real error) / 125
    ≈ 13/125 × real_error
    ≈ 0.10 × real_error
```

The model can't efficiently learn the variable part because:
1. **Gradient dilution**: Gradients from 13 informative dims are averaged with near-zero gradients from 112 constant dims
2. **Learning rate is wrong**: The effective LR for informative dims is ~10× too low
3. **The test set has only 1 sample** (5 pairs, 80/20 split → 4 train, 1 test) — test MSE is just noise from one sample

## Why More Parameters Didn't Help

The task is **not capacity-limited**, it's **signal-limited**:
- The model has plenty of capacity to memorize the constant background
- The actual regression task (predicting 13 variable dims from 143 input dims with only 4–5 training pairs) is **extremely data-starved**
- More parameters = more overfitting on the tiny useful signal, no improvement on test

## Additional Issues Found

### 1. Catastrophically Small Dataset
```
Total pairs: 5
Training pairs: 4 (80%)
Testing pairs: 1 (20%)
```
You're training a 1.5M parameter model on **4 training examples** (augmented to 200 via noise). This isn't supervised learning — this is memorization with noise.

### 2. Augmentation Creates Fake Copies, Not Real Data
The augmentation adds Gaussian noise to **observations** (inputs) but keeps **targets identical**. This means 50 copies of the same 4 targets with slightly different inputs. The model learns to ignore input variation and output the mean target.

### 3. Gain Head Initialization Partially Broken

In [_init_identity_bias](file:///home/itorousa/Documents/Code/faurge/kaggle/ursula/04_define_ursula/04_cell_3.py#L63-L67):
```python
nn.init.zeros_(gain_head.weight)  # ← zeroes ALL weights
nn.init.zeros_(gain_head.bias)
```
The gain head starts with **all-zero weights**, meaning it always outputs `tanh(0) = 0` initially. While gradients will eventually break the symmetry, this slows learning for the gain parameter.

### 4. Mismatched Architecture Definitions

| File | OUTPUT_DIM | Plugin Heads | Hidden Dim |
|------|-----------|-------------|------------|
| [agents/ursula.py](file:///home/itorousa/Documents/Code/faurge/agents/ursula.py) | **227** | 7 plugins (EQ+Comp+Esser+Sat+Lim+Trans+Gain) | 512 |
| [04_cell_1.py](file:///home/itorousa/Documents/Code/faurge/kaggle/ursula/04_define_ursula/04_cell_1.py) (Kaggle) | **125** | 2 plugins (EQ+Gain) | 512 |

The Kaggle training uses the **125D / 2-plugin** version, but the local `agents/ursula.py` defines the **227D / 7-plugin** version. The training code is correct (it imports from the exported Kaggle version), but this divergence will cause problems when you try to load weights locally.

## Recommended Fixes

### Fix 1: Redesign the Target Representation (Critical)

**Don't encode unused EQ bands at all.** Instead, use a sparse representation:

```python
# Option A: Variable-length target with mask
# - 31 active flags (binary: is this band active?)
# - Per-active-band params (freq, gain, q, type)
# - Global gain

# Option B: Only predict the ACTIVE bands
# - Degradation tells you which bands were used
# - Model only predicts those band params + global gain
# - Loss only on active dimensions
```

### Fix 2: Use a Loss Mask (Quick Fix)

```python
# In training loop:
mask = torch.ones(OUTPUT_DIM, device=DEVICE)
# Zero out constant/identity dimensions 
for b in range(31):
    if b >= n_active_bands:  # identity band
        idx = b * 4
        mask[idx:idx+4] = 0.0  # don't penalize identity bands

loss = (mask * (pred - target) ** 2).sum() / mask.sum()
```

### Fix 3: Get WAY More Data (Critical)

4 training pairs is absurdly small. You need at minimum **hundreds** of pairs, ideally **thousands**. The degradation generation pipeline already exists — just generate more data.

### Fix 4: Simplify the Output

Since most bands are identity, consider predicting:
- **31 gain values** (one per EQ band) — identity = 0
- **1 global gain** — identity = 0
- Total: 32D output, ALL dimensions are informative

Keep freq/Q/filter_type fixed at sensible defaults (bark-scale frequencies, Q=1, peak filters). The model only needs to learn **how much to boost/cut each band**.

> [!IMPORTANT]
> Fix 3 (more data) and Fix 4 (simplified output) together would likely solve the problem entirely. Fix 1 or 2 would help but won't overcome the 4-sample dataset limit.
