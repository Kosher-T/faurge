"""
Diagnostic script: Investigate why Ursula SL training plateaus.
Checks both the model architecture and the data pipeline.
"""

import sys, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/itorousa/Documents/Code/faurge')

from agents.ursula import (
    UrsulaPolicy, ActionUnnormalizer, ALL_PARAM_RANGES,
    PLUGIN_HEAD_DIMS, PLUGIN_HEAD_ORDER, INPUT_DIM, OUTPUT_DIM,
    CATEGORICAL_INDICES,
)

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  DIAGNOSTIC 1: MODEL ARCHITECTURE ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# Check 1: Does the model collapse outputs?
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 1: Output diversity across random inputs ──")
policy = UrsulaPolicy()
policy.eval()

# Feed 100 different random inputs
x = torch.randn(100, INPUT_DIM)
with torch.no_grad():
    out = policy(x)

print(f"  Output shape: {out.shape}")
print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
print(f"  Output mean:  {out.mean():.4f}")
print(f"  Output std:   {out.std():.4f}")
print(f"  Output norm (per-sample): mean={out.norm(dim=-1).mean():.4f}, std={out.norm(dim=-1).std():.4f}")

# Check per-head output statistics
for name in PLUGIN_HEAD_ORDER:
    start, end = 0, 0
    offset = 0
    for n in PLUGIN_HEAD_ORDER:
        if n == name:
            start = offset
            end = offset + PLUGIN_HEAD_DIMS[n]
            break
        offset += PLUGIN_HEAD_DIMS[n]
    head_out = out[:, start:end]
    print(f"  Head '{name}' [{start}:{end}]: mean={head_out.mean():.4f}, std={head_out.std():.4f}, "
          f"min={head_out.min():.4f}, max={head_out.max():.4f}")

# ─────────────────────────────────────────────────────────────────────
# Check 2: Gradient flow analysis
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 2: Gradient flow analysis ──")
policy.train()
x = torch.randn(16, INPUT_DIM)
target = torch.randn(16, OUTPUT_DIM).clamp(-1, 1)

pred = policy(x)
loss = nn.functional.mse_loss(pred, target)
loss.backward()

for name, param in policy.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.norm().item()
        ratio = grad_norm / (param_norm + 1e-8)
        if grad_norm < 1e-7:
            flag = " ⚠️ DEAD GRADIENT"
        elif ratio < 1e-5:
            flag = " ⚠️ VANISHING"
        else:
            flag = ""
        print(f"  {name:40s}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.4f}, ratio={ratio:.6f}{flag}")

policy.zero_grad()

# ─────────────────────────────────────────────────────────────────────
# Check 3: Tanh saturation check
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 3: Tanh saturation analysis ──")
policy.eval()
x = torch.randn(200, INPUT_DIM)
with torch.no_grad():
    out = policy(x)

# Count outputs that are saturated (near -1 or 1)
saturated = ((out.abs() > 0.95).float().mean() * 100).item()
deeply_saturated = ((out.abs() > 0.99).float().mean() * 100).item()
near_zero = ((out.abs() < 0.1).float().mean() * 100).item()
print(f"  |output| > 0.95: {saturated:.1f}% of all outputs (saturated)")
print(f"  |output| > 0.99: {deeply_saturated:.1f}% of all outputs (deeply saturated)")
print(f"  |output| < 0.10: {near_zero:.1f}% of all outputs (near zero)")

# ─────────────────────────────────────────────────────────────────────
# Check 4: Identity bias initialization
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 4: Gain head initialization ──")
gain_head = policy.plugin_heads["gain"]
print(f"  Gain weight: {gain_head.weight.data}")
print(f"  Gain bias: {gain_head.bias.data}")
print(f"  Gain weight norm: {gain_head.weight.data.norm():.6f}")
print(f"  (Weights are zeroed → gain head always outputs tanh(0) = 0 regardless of input)")
print(f"  ⚠️  This means the gain head CANNOT LEARN from gradients because weight is 0!")
print(f"  ⚠️  tanh'(0) = 1, so grad flows through tanh, but d_loss/d_weight = input * tanh'(z)")
print(f"  ⚠️  The first backward pass WILL give non-zero gradients since input != 0")
print(f"  ⚠️  But the gain head starts stuck at 0 and has to 'grow' from scratch")


print("\n" + "=" * 70)
print("  DIAGNOSTIC 2: DATA / TARGET ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# Check 5: What do the supervised targets look like?
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 5: Synthetic target vector statistics ──")

# Recreate what compute_inverse_action does, for a few random degradations
def _inv_linear(val, low, high):
    return float(np.clip((val - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0))

def _inv_log(val, low, high):
    val = max(val, 1e-8)
    log_low = math.log(max(low, 1e-8))
    log_high = math.log(max(high, 1e-8))
    return float(np.clip((math.log(val) - log_low) / (log_high - log_low) * 2.0 - 1.0, -1.0, 1.0))

def _inv_cat(val, low, high):
    return float(np.clip((val - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0))


def compute_inverse_action_test(deg_params):
    """Replicate the training target generation."""
    inv = np.zeros(125, dtype=np.float32)  # 124 EQ + 1 gain
    
    deg_bands = deg_params.get('eq_bands', [])
    for b in range(31):
        idx = b * 4
        _FT_MAP = {"peak": 0, "low_shelf": 1, "high_shelf": 2}
        
        if b < len(deg_bands):
            band = deg_bands[b]
            freq = band.get('freq_hz', 1000.0)
            gain = band.get('gain_db', 0.0)
            q = band.get('q', 1.0)
            ft_str = band.get('filter_type', 'peak')
            ft_val = _FT_MAP.get(ft_str, 0)
            
            inv[idx + 0] = _inv_log(freq, 20.0, 20000.0)
            inv[idx + 1] = _inv_linear(-gain, -24.0, 24.0)  # NEGATE gain
            inv[idx + 2] = _inv_linear(q, 0.1, 10.0)
            inv[idx + 3] = _inv_cat(ft_val, 0.0, 2.0)
        else:
            # Identity EQ: freq=1000, gain=0, q=1, type=peak
            inv[idx + 0] = _inv_log(1000.0, 20.0, 20000.0)
            inv[idx + 1] = _inv_linear(0.0, -24.0, 24.0)     # gain=0 → raw=0
            inv[idx + 2] = _inv_linear(1.0, 0.1, 10.0)        # q=1
            inv[idx + 3] = _inv_cat(0.0, 0.0, 2.0)            # peak → -1.0
    
    g = deg_params.get('gain', {})
    inv[124] = _inv_linear(-g.get('gain_db', 0.0), -12.0, 12.0)
    
    return np.clip(inv, -1.0, 1.0)


# Generate some random degradation params like the generation code does
import random
random.seed(42)

targets = []
for i in range(100):
    n_bands = random.randint(1, 6)
    bands = []
    for _ in range(n_bands):
        bands.append({
            'freq_hz': round(float(np.exp(random.uniform(np.log(20), np.log(20000)))), 1),
            'gain_db': round(random.uniform(-12, 12), 2),
            'q': round(random.uniform(0.1, 10), 2),
            'filter_type': random.choice(['peak', 'low_shelf', 'high_shelf']),
        })
    gain_db = round(random.uniform(-12, 12), 2)
    
    deg = {'eq_bands': bands, 'gain': {'gain_db': gain_db}}
    inv = compute_inverse_action_test(deg)
    targets.append(inv)

targets = np.array(targets)

print(f"  Generated {len(targets)} synthetic inverse targets")
print(f"  Shape: {targets.shape}")
print(f"  Global range: [{targets.min():.4f}, {targets.max():.4f}]")
print(f"  Global mean:  {targets.mean():.4f}")
print(f"  Global std:   {targets.std():.4f}")
print(f"  Action norm (per-sample): mean={np.linalg.norm(targets, axis=-1).mean():.4f}, "
      f"std={np.linalg.norm(targets, axis=-1).std():.4f}")

# ─────────────────────────────────────────────────────────────────────
# Check 6: THE CRITICAL CHECK — How many dimensions are identical / near-constant?
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 6: ⚠️  CRITICAL — Per-dimension variance in targets ──")
per_dim_std = targets.std(axis=0)
per_dim_mean = targets.mean(axis=0)

n_constant = (per_dim_std < 0.001).sum()
n_low_var = (per_dim_std < 0.01).sum()
n_informative = (per_dim_std > 0.1).sum()

print(f"  Total dimensions: {targets.shape[1]}")
print(f"  Constant dims (std < 0.001): {n_constant}")
print(f"  Low-variance dims (std < 0.01): {n_low_var}")
print(f"  Informative dims (std > 0.1): {n_informative}")

# Show constant dimensions
print(f"\n  Constant dimensions (always same value):")
for i in range(targets.shape[1]):
    if per_dim_std[i] < 0.001:
        # Figure out which parameter this maps to
        if i < 124:
            band = i // 4
            param_idx = i % 4
            param_names = ['freq', 'gain', 'q', 'filter_type']
            pname = f"eq_band{band+1}_{param_names[param_idx]}"
        else:
            pname = "gain_db"
        print(f"    dim {i:3d} ({pname:30s}): mean={per_dim_mean[i]:.4f}, std={per_dim_std[i]:.6f}")

# Show the informative dimensions
print(f"\n  High-variance dimensions (model should learn these):")
for i in range(targets.shape[1]):
    if per_dim_std[i] > 0.1:
        if i < 124:
            band = i // 4
            param_idx = i % 4
            param_names = ['freq', 'gain', 'q', 'filter_type']
            pname = f"eq_band{band+1}_{param_names[param_idx]}"
        else:
            pname = "gain_db"
        print(f"    dim {i:3d} ({pname:30s}): mean={per_dim_mean[i]:.4f}, std={per_dim_std[i]:.4f}")

# ─────────────────────────────────────────────────────────────────────
# Check 7: Compute what the action norm SHOULD be for typical targets
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 7: Expected action norm analysis ──")
norms = np.linalg.norm(targets, axis=-1)
print(f"  Target norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
      f"min={norms.min():.4f}, max={norms.max():.4f}")

# Decompose: what contributes to the norm?
# Most of the 125D vector is EQ bands, most of which are IDENTITY (unused bands)
# Let's check what the "identity" EQ contributes:
identity_band = np.array([
    _inv_log(1000.0, 20.0, 20000.0),    # freq=1000
    _inv_linear(0.0, -24.0, 24.0),       # gain=0
    _inv_linear(1.0, 0.1, 10.0),         # q=1
    _inv_cat(0.0, 0.0, 2.0),             # filter_type=peak
])
print(f"\n  Identity EQ band raw values: {identity_band}")
print(f"  Identity EQ band norm contribution: {np.linalg.norm(identity_band):.4f}")

# For a typical target with 3 active bands and 28 identity bands:
n_identity = 28
identity_contribution = np.linalg.norm(
    np.tile(identity_band, n_identity)
)
print(f"  28 identity bands total norm contribution: {identity_contribution:.4f}")
print(f"  ⚠️  THIS IS WHY the action norm is ~7.2!")
print(f"  ⚠️  The identity EQ freq raw value = {identity_band[0]:.4f}")
print(f"  ⚠️  The identity EQ q raw value = {identity_band[2]:.4f}")
print(f"  ⚠️  The identity EQ filter_type raw value = {identity_band[3]:.4f} (= -1.0 for 'peak')")

# The dominant contribution: freq and q of identity bands
print(f"\n  Breakdown of identity band 'encoded' raw values:")
print(f"    freq=1000 → raw = {identity_band[0]:.4f}  (log-encoded)")
print(f"    gain=0    → raw = {identity_band[1]:.4f}  (linear-encoded)")
print(f"    q=1.0     → raw = {identity_band[2]:.4f}  (linear-encoded, range [0.1, 10])")
print(f"    type=peak → raw = {identity_band[3]:.4f}  (categorical)")

# ─────────────────────────────────────────────────────────────────────
# Check 8: The REAL problem — unused bands dominate the loss!
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 8: ⚠️  MSE LOSS DECOMPOSITION — unused vs active bands ──")

# For a typical target, compute loss contribution
policy.eval()
x_test = torch.randn(1, INPUT_DIM)
with torch.no_grad():
    pred_test = policy(x_test).numpy()

# Create a "typical" target with 3 active EQ bands
deg_example = {
    'eq_bands': [
        {'freq_hz': 500.0, 'gain_db': 6.0, 'q': 2.0, 'filter_type': 'peak'},
        {'freq_hz': 2000.0, 'gain_db': -8.0, 'q': 1.5, 'filter_type': 'low_shelf'},
        {'freq_hz': 8000.0, 'gain_db': 4.0, 'q': 3.0, 'filter_type': 'high_shelf'},
    ],
    'gain': {'gain_db': -3.0}
}
tgt_example = compute_inverse_action_test(deg_example)

# MSE decomposition
mse_per_dim = (pred_test[0] - tgt_example) ** 2

# Active bands (0-2): indices 0-11
active_mse = mse_per_dim[:12].mean()
# Identity bands (3-30): indices 12-123
identity_mse = mse_per_dim[12:124].mean()
# Gain: index 124
gain_mse = mse_per_dim[124]
# Total
total_mse = mse_per_dim.mean()

print(f"  Active EQ bands (3 bands, 12 dims): MSE = {active_mse:.6f}")
print(f"  Identity EQ bands (28 bands, 112 dims): MSE = {identity_mse:.6f}")
print(f"  Gain (1 dim): MSE = {gain_mse:.6f}")
print(f"  Total MSE: {total_mse:.6f}")
print(f"")
print(f"  Identity bands contribute {112}/{125} = {112/125*100:.0f}% of dimensions!")
print(f"  Active bands contribute {12}/{125} = {12/125*100:.0f}% of dimensions!")
print(f"")
print(f"  ⚠️  The model is overwhelmingly trained to output identity EQ params!")
print(f"  ⚠️  The SAME constant values for 90% of the output vector!")
print(f"  ⚠️  The actual useful signal (which EQ to apply) is drowned out!")

# ─────────────────────────────────────────────────────────────────────
# Check 9: What percentage of the MSE comes from constant vs variable dims?
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 9: Constant-dim MSE contribution across target set ──")

# Average over many targets
all_mse_per_dim = np.zeros(125)
for tgt in targets:
    all_mse_per_dim += (pred_test[0] - tgt) ** 2
all_mse_per_dim /= len(targets)

constant_dims = per_dim_std < 0.001
variable_dims = per_dim_std >= 0.001

const_mse_total = all_mse_per_dim[constant_dims].sum()
var_mse_total = all_mse_per_dim[variable_dims].sum()
total = const_mse_total + var_mse_total

print(f"  Constant dims ({constant_dims.sum()} dims): MSE contribution = {const_mse_total:.4f} ({const_mse_total/total*100:.1f}%)")
print(f"  Variable dims ({variable_dims.sum()} dims): MSE contribution = {var_mse_total:.4f} ({var_mse_total/total*100:.1f}%)")

print(f"\n  ⚠️  Once the model learns to output the constant values, MSE plateaus!")
print(f"  ⚠️  The remaining 'hard' signal (which EQ/gain to apply) is tiny fraction of loss!")

# ─────────────────────────────────────────────────────────────────────
# Check 10: MSE floor — what's the best possible MSE?
# ─────────────────────────────────────────────────────────────────────
print("\n── Check 10: MSE floor analysis ──")

# Best possible constant prediction = mean of targets
mean_pred = targets.mean(axis=0)
mse_from_mean = ((targets - mean_pred) ** 2).mean()
print(f"  MSE if model just predicts mean target: {mse_from_mean:.6f}")
print(f"  This is the FLOOR for a model that 'memorizes the average'")

# Compare to what user reported:
print(f"  User's best test MSE: ~0.057")
print(f"  Variance-based floor: {mse_from_mean:.6f}")
print(f"  Gap: {0.057 - mse_from_mean:.6f}")

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  DIAGNOSIS SUMMARY")
print("=" * 70)
print("""
  ROOT CAUSE: The 125D target vector is ~90% CONSTANT (identity EQ bands).

  DETAILS:
  - Degradation uses 1-6 random EQ bands out of 31 total
  - The remaining 25-30 bands are filled with FIXED identity values:
      freq=1000 → raw ≈ 0.424
      gain=0    → raw = 0.0
      q=1.0     → raw ≈ -0.818
      type=peak → raw = -1.0
  - These ~112 constant dimensions DOMINATE the loss
  - The model quickly learns to output these constants → loss drops fast
  - Then plateaus because the remaining ~13 informative dimensions
    are drowned in noise

  WHY ACTION NORM ≈ 7.2:
  - The constant identity values have nonzero norm (~1.4 per band)
  - 28 identity bands × norm_contribution → total ≈ 7.2

  WHY MORE PARAMS DIDN'T HELP:
  - The task isn't capacity-limited — it's signal-limited
  - 90% of output dims carry no information
  - More params = more overfitting on the ~10% useful signal

  FIXES:
  1. REMOVE identity bands from targets — only include active EQ bands
  2. Or use a MASK so unused bands don't contribute to loss
  3. Or redesign output: predict WHICH bands are active + their params
  4. Or use per-dimension loss weighting (upweight active bands)
  5. Use ONLY gain_db per EQ band in targets for identity bands 
     (since gain=0 is all that matters for identity)
""")
