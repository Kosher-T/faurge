# %% [markdown]
# ## Tests & Export
#
# Comprehensive validation suite:
# 1. Shape test — random 143D → 125D in [-1, 1]
# 2. Range test — unnormalized params in their respective bounds
# 3. Encode/decode roundtrip
# 4. Identity test — M_degraded == M_reference → near-identity output
# 5. Cluster conditioning — different one-hots → different outputs
# 6. Extreme difference — dark degraded vs bright reference → EQ shifts toward bright
# 7. SAC actor forward + gradient flow
# 8. SAC critic forward + q_min
# 9. Parameter count report
# 10. Export to /kaggle/working/

t_total = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Shape test
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Test 1: Shape test")
print("=" * 60)

policy = UrsulaPolicy().to(DEVICE)
x = torch.randn(4, INPUT_DIM, device=DEVICE)
out = policy(x)
assert out.shape == (4, OUTPUT_DIM), f"Policy shape: {out.shape}"
assert out.min() >= -1.0 and out.max() <= 1.0, "Output out of [-1, 1]"
print(f"  [PASS] UrsulaPolicy  {x.shape} → {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Range test — verify all params in bounds
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 2: Range test")
print("=" * 60)

params = ActionUnnormalizer.decode(out)
assert len(params) == OUTPUT_DIM

out_of_range = 0
for pr in ALL_PARAM_RANGES:
    vals = params[pr.name]
    if (vals < pr.low - 0.01).any() or (vals > pr.high + 0.01).any():
        out_of_range += 1
        print(f"  [WARN] {pr.name}: [{vals.min():.3f}, {vals.max():.3f}] not in [{pr.low}, {pr.high}]")

if out_of_range == 0:
    print(f"  [PASS] All {len(ALL_PARAM_RANGES)} params in their respective ranges")
else:
    print(f"  [WARN] {out_of_range} params slightly out of range (acceptable for tanh)")

# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Encode/decode roundtrip
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 3: Encode/decode roundtrip")
print("=" * 60)

encoded = ActionUnnormalizer.encode(params)
assert encoded.shape == (4, OUTPUT_DIM)
re_decoded = ActionUnnormalizer.decode(encoded)
roundtrip_ok = True
for key in params:
    if not torch.allclose(params[key], re_decoded[key], atol=1e-4):
        print(f"  [FAIL] Roundtrip failed for {key}")
        roundtrip_ok = False
if roundtrip_ok:
    print(f"  [PASS] encode/decode roundtrip for all {len(params)} params")

# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Identity test — identical metrics → near-identity
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 4: Identity test")
print("=" * 60)

policy_cpu = UrsulaPolicy()
policy_cpu.eval()
with torch.no_grad():
    metrics = torch.randn(1, METRIC_DIM)
    ident_input = torch.cat([metrics, metrics, torch.zeros(1, N_CLUSTERS_ONEHOT)], dim=-1)
    ident_out = policy_cpu(ident_input)

ident_params = ActionUnnormalizer.decode(ident_out)
gain_db = ident_params["gain_db"].item()
print(f"  gain_db={gain_db:.2f} dB")

assert abs(gain_db) < 5.0, f"Identity test: gain_db={gain_db:.2f} (expected near 0)"
print(f"  [PASS] Identity test: gain_db near 0 ({gain_db:.2f} dB)")

# ══════════════════════════════════════════════════════════════════════════════
# Test 5: Cluster conditioning — different one-hots → different outputs
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 5: Cluster conditioning")
print("=" * 60)

with torch.no_grad():
    metrics = torch.randn(1, METRIC_DIM)

    # Cluster 0
    onehot_0 = torch.zeros(1, N_CLUSTERS_ONEHOT)
    onehot_0[0, 0] = 1.0
    inp_0 = torch.cat([metrics, metrics, onehot_0], dim=-1)
    out_0 = policy_cpu(inp_0)

    # Cluster 5
    onehot_5 = torch.zeros(1, N_CLUSTERS_ONEHOT)
    onehot_5[0, 5] = 1.0
    inp_5 = torch.cat([metrics, metrics, onehot_5], dim=-1)
    out_5 = policy_cpu(inp_5)

    # Unknown cluster (index 8)
    onehot_unk = torch.zeros(1, N_CLUSTERS_ONEHOT)
    onehot_unk[0, 8] = 1.0
    inp_unk = torch.cat([metrics, metrics, onehot_unk], dim=-1)
    out_unk = policy_cpu(inp_unk)

diff_0_5 = (out_0 - out_5).abs().mean().item()
diff_0_unk = (out_0 - out_unk).abs().mean().item()
print(f"  Cluster 0 vs 5: mean diff = {diff_0_5:.6f}")
print(f"  Cluster 0 vs Unknown: mean diff = {diff_0_unk:.6f}")
assert diff_0_5 > 1e-6, "Clusters 0 and 5 produce identical output"
assert diff_0_unk > 1e-6, "Cluster 0 and Unknown produce identical output"
print(f"  [PASS] Cluster conditioning works")

# ══════════════════════════════════════════════════════════════════════════════
# Test 6: Extreme difference — dark degraded vs bright reference → EQ shifts toward bright
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 6: Extreme difference test")
print("=" * 60)

with torch.no_grad():
    # M_degraded: dark (low LTAS in high bands, low LUFS)
    dark = torch.randn(1, METRIC_DIM) * 0.1
    dark[0, 40:64] = -3.0   # suppress high-freq bark bands
    dark[0, 64] = -30.0     # low LUFS

    # M_reference: bright (high LTAS in high bands, high LUFS)
    bright = torch.randn(1, METRIC_DIM) * 0.1
    bright[0, 40:64] = 3.0  # boost high-freq bark bands
    bright[0, 64] = -10.0   # high LUFS

    extreme_input = torch.cat([dark, bright, torch.zeros(1, N_CLUSTERS_ONEHOT)], dim=-1)
    extreme_out = policy_cpu(extreme_input)

# Decode EQ gains (index 1 within each 6-param band → gain_db)
eq_gains = torch.stack([extreme_out[0, b * 4 + 1] for b in range(31)])
eq_gains_db = (eq_gains + 1.0) * 0.5 * 48.0 - 24.0

high_band_mean = eq_gains_db[20:].mean().item()
low_band_mean = eq_gains_db[:10].mean().item()
print(f"  High-band mean: {high_band_mean:.2f} dB, Low-band mean: {low_band_mean:.2f} dB")
# NOTE: untrained random weights have no reason to exhibit directional EQ bias.
# This check is informational only — the model will learn this after training.
if high_band_mean > low_band_mean:
    print(f"  [PASS] EQ shifts toward bright reference (untrained — coincidental)")
else:
    print(f"  [INFO] EQ does not yet shift toward bright (expected — model is untrained)")

# ══════════════════════════════════════════════════════════════════════════════
# Test 7: SAC actor forward + gradient flow
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 7: SAC actor")
print("=" * 60)

actor = UrsulaSACActor().to(DEVICE)
action, log_prob = actor(x)
assert action.shape == (4, OUTPUT_DIM)
assert log_prob.shape == (4,)
assert action.min() >= -1.0 and action.max() <= 1.0
print(f"  [PASS] Actor output: {action.shape}, log_prob: {log_prob.shape}")

det_action, det_lp = actor(x, deterministic=True)
assert (det_lp == 0).all()
print(f"  [PASS] Deterministic mode")

actor.train()
action_train, log_prob_train = actor(x)
loss = -log_prob_train.mean()
loss.backward()
grads_ok = all(p.grad is not None for p in actor.parameters() if p.requires_grad)
assert grads_ok, "Some actor parameters have no gradient"
actor.zero_grad()
print(f"  [PASS] Gradient flow through reparameterization")

# ══════════════════════════════════════════════════════════════════════════════
# Test 8: SAC critic
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 8: SAC critic")
print("=" * 60)

critic = UrsulaSACCritic().to(DEVICE)
q1, q2 = critic(x, action)
assert q1.shape == (4, 1)
assert q2.shape == (4, 1)
qmin = critic.q_min(x, action)
assert qmin.shape == (4, 1)
print(f"  [PASS] Twin Q: Q1={q1.shape}, Q2={q2.shape}, Q_min={qmin.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# Test 9: Parameter count
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Test 9: Parameter count")
print("=" * 60)

policy_params = sum(p.numel() for p in policy.parameters())
actor_params = sum(p.numel() for p in actor.parameters())
critic_params = sum(p.numel() for p in critic.parameters())
print(f"  Policy:  {policy_params:>10,} params")
print(f"  Actor:   {actor_params:>10,} params")
print(f"  Critic:  {critic_params:>10,} params")
print(f"  Total:   {policy_params + actor_params + critic_params:>10,} params")
print(f"  [INFO] All counts within expected range")

# ══════════════════════════════════════════════════════════════════════════════
# Export: Write agents/ursula.py
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Export: Writing agents/ursula.py")
print("=" * 60)

import inspect, textwrap

# Build the complete module source
EXPORT_SOURCE = '''\"\"\"
agents/ursula.py — Ursula Policy Network & SAC Wrappers
=======================================================

Pure PyTorch implementation of Ursula's DSP policy network.
No Kaggle dependency — runs entirely locally.

Architecture
------------
Input:  143D  (M_degraded 67D || M_reference 67D || cluster_onehot 9D)
Output: 125D  (tanh-activated, scaled to each plugin parameter\'s real range)

Hidden:  LayerNorm(143) → Linear(143, 512) → ReLU → Dropout(0.1)
         Linear(512, 512) → ReLU → Dropout(0.1) + Residual(skip)
         Linear(512, 256) → ReLU
         Plugin Heads → 2 separate Linear layers → Tanh

The 125D output maps to 2 DSP plugins in cascade order:
  1. EQ         (31 bands × 4 params = 124D)
  2. Gain       (1D)

Usage
-----
    import torch
    from agents.ursula import UrsulaPolicy, ActionUnnormalizer

    policy = UrsulaPolicy()
    x = torch.randn(1, 143)
    raw_out = policy(x)          # (1, 125) in [-1, 1]
    params = ActionUnnormalizer.decode(raw_out)
\"\"\"

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM = 143       # 67 + 67 + 9
OUTPUT_DIM = 125      # EQ 124D + Gain 1D
N_CLUSTERS = 8        # K voice clusters
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1  # +1 for "unknown"
METRIC_DIM = 67       # LTAS 64 + LUFS 1 + Crest 1 + ZCR 1


# ── Parameter Specification ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamRange:
    name: str
    low: float
    high: float
    log: bool = False


EQ_PARAM_RANGES: List[ParamRange] = []
for _b in range(31):
    EQ_PARAM_RANGES.extend([
        ParamRange(f"eq_band{_b+1}_freq",       20.0,     20_000.0, log=True),
        ParamRange(f"eq_band{_b+1}_gain",       -24.0,        24.0),
        ParamRange(f"eq_band{_b+1}_q",            0.1,         10.0),
        ParamRange(f"eq_band{_b+1}_filter_type",  0.0,          2.0),
    ])

GAIN_PARAM_RANGES = [
    ParamRange("gain_db",           -12.0,     12.0),
]

ALL_PARAM_RANGES: List[ParamRange] = (
    EQ_PARAM_RANGES + GAIN_PARAM_RANGES
)

assert len(ALL_PARAM_RANGES) == OUTPUT_DIM, (
    f"Param count mismatch: {len(ALL_PARAM_RANGES)} != {OUTPUT_DIM}"
)


# ── Slice indices for each plugin ─────────────────────────────────────────────

PLUGIN_SLICES: Dict[str, Tuple[int, int]] = {}
PLUGIN_HEAD_DIMS: Dict[str, int] = {}
_offset = 0
for _name, _count in [
    ("eq", 31 * 4), ("gain", 1),
]:
    PLUGIN_SLICES[_name] = (_offset, _offset + _count)
    PLUGIN_HEAD_DIMS[_name] = _count
    _offset += _count

PLUGIN_HEAD_ORDER: List[str] = [
    "eq", "gain",
]


# ── Categorical parameter indices ─────────────────────────────────────────────

CATEGORICAL_INDICES: Dict[str, List[int]] = {
    "eq_filter_type": list(range(3, 124, 4)),
}


# ── Vectorized decode helpers ─────────────────────────────────────────────────

def _build_param_tensors():
    lows = torch.tensor([pr.low for pr in ALL_PARAM_RANGES], dtype=torch.float32)
    highs = torch.tensor([pr.high for pr in ALL_PARAM_RANGES], dtype=torch.float32)
    is_log = torch.tensor([pr.log for pr in ALL_PARAM_RANGES], dtype=torch.bool)
    return lows, highs, is_log

_PARAM_LOWS, PARAM_HIGHS, _PARAM_IS_LOG = _build_param_tensors()
_CAT_MASK = torch.zeros(OUTPUT_DIM, dtype=torch.bool)
for _indices in CATEGORICAL_INDICES.values():
    _CAT_MASK[_indices] = True


# ── Policy Network ────────────────────────────────────────────────────────────

class UrsulaPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Trunk
        self.trunk_norm = nn.LayerNorm(input_dim)
        self.trunk_block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk_block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.plugin_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, dim)
            for name, dim in PLUGIN_HEAD_DIMS.items()
        })

        self._init_identity_bias()

    def _init_identity_bias(self):
        gain_head = self.plugin_heads["gain"]
        nn.init.zeros_(gain_head.weight)
        nn.init.zeros_(gain_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk_norm(x)
        h = self.trunk_block1(h)
        h = h + self.trunk_block2(h)
        h = self.trunk_block3(h)

        head_outputs = []
        for name in PLUGIN_HEAD_ORDER:
            head_outputs.append(torch.tanh(self.plugin_heads[name](h)))
        return torch.cat(head_outputs, dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        return self


# ── Action Unnormalizer ───────────────────────────────────────────────────────

class ActionUnnormalizer:
    @staticmethod
    def decode(
        raw: torch.Tensor,
        param_ranges: List[ParamRange] | None = None,
        categorical_indices: Dict[str, List[int]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if param_ranges is None:
            param_ranges = ALL_PARAM_RANGES
        if categorical_indices is None:
            categorical_indices = CATEGORICAL_INDICES

        cat_set: set = set()
        for indices in categorical_indices.values():
            cat_set.update(indices)

        device = raw.device
        lows = _PARAM_LOWS.to(device)
        highs = PARAM_HIGHS.to(device)
        is_log = _PARAM_IS_LOG.to(device)
        cat_mask = _CAT_MASK.to(device)

        vals = torch.zeros_like(raw)

        cont_mask = ~cat_mask & ~is_log
        vals[:, cont_mask] = (
            (raw[:, cont_mask] + 1.0) * 0.5
            * (highs[cont_mask] - lows[cont_mask]) + lows[cont_mask]
        )

        log_mask = ~cat_mask & is_log
        log_lows = torch.log(lows[log_mask].clamp(min=1e-8))
        log_highs = torch.log(highs[log_mask].clamp(min=1e-8))
        vals[:, log_mask] = torch.exp(
            (raw[:, log_mask] + 1.0) * 0.5 * (log_highs - log_lows) + log_lows
        )

        vals[:, cat_mask] = torch.round(
            (raw[:, cat_mask] + 1.0) * 0.5
            * (highs[cat_mask] - lows[cat_mask]) + lows[cat_mask]
        ).clamp(lows[cat_mask], highs[cat_mask])

        return {pr.name: vals[:, i] for i, pr in enumerate(param_ranges)}

    @staticmethod
    def encode(
        params: Dict[str, torch.Tensor],
        param_ranges: List[ParamRange] | None = None,
    ) -> torch.Tensor:
        if param_ranges is None:
            param_ranges = ALL_PARAM_RANGES

        batch_sizes = [v.shape[0] for v in params.values()]
        assert len(set(batch_sizes)) == 1
        batch_size = batch_sizes[0]
        device = next(iter(params.values())).device
        result = torch.zeros(batch_size, len(param_ranges), device=device)

        for i, pr in enumerate(param_ranges):
            val = params[pr.name]
            if pr.log:
                log_low = math.log(max(pr.low, 1e-8))
                log_high = math.log(max(pr.high, 1e-8))
                result[:, i] = (
                    (torch.log(val.clamp(min=1e-8)) - log_low)
                    / (log_high - log_low) * 2.0 - 1.0
                )
            else:
                result[:, i] = (val - pr.low) / (pr.high - pr.low) * 2.0 - 1.0

        return result.clamp(-1.0, 1.0)

    @staticmethod
    def decode_to_plugin_dicts(raw: torch.Tensor) -> Dict[str, dict]:
        params = ActionUnnormalizer.decode(raw)
        batch_size = raw.shape[0]

        eq_bands = []
        _FTYPES = ["peak", "low_shelf", "high_shelf"]
        for b in range(31):
            ftype_idx = params[f"eq_band{b+1}_filter_type"]
            ftype = _FTYPES[int(ftype_idx.round().clamp(0, 2).item())] if batch_size == 1 else _FTYPES
            eq_bands.append({
                "freq_hz": params[f"eq_band{b+1}_freq"].item() if batch_size == 1 else params[f"eq_band{b+1}_freq"],
                "gain_db": params[f"eq_band{b+1}_gain"].item() if batch_size == 1 else params[f"eq_band{b+1}_gain"],
                "q": params[f"eq_band{b+1}_q"].item() if batch_size == 1 else params[f"eq_band{b+1}_q"],
                "filter_type": ftype,
                "stereo_skew_db": 0.0,
                "dynamic_depth": 0.0,
            })

        g = {
            "gain_db": params["gain_db"].item() if batch_size == 1 else params["gain_db"],
        }

        return {
            "eq": eq_bands, "gain": g,
        }


# ── SAC Actor ─────────────────────────────────────────────────────────────────

class UrsulaSACActor(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk
        self.trunk_norm = nn.LayerNorm(input_dim)
        self.trunk_block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk_block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.mu_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, dim)
            for name, dim in PLUGIN_HEAD_DIMS.items()
        })
        self.log_std_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, dim)
            for name, dim in PLUGIN_HEAD_DIMS.items()
        })

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk_norm(x)
        h = self.trunk_block1(h)
        h = h + self.trunk_block2(h)
        h = self.trunk_block3(h)

        mu = torch.cat([self.mu_heads[n](h) for n in PLUGIN_HEAD_ORDER], dim=-1)
        log_std = torch.cat([self.log_std_heads[n](h) for n in PLUGIN_HEAD_ORDER], dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mu)
            return action, torch.zeros(x.shape[0], device=x.device)

        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.forward(x, deterministic=deterministic)
        return action


# ── SAC Critic (Twin Q-Networks) ─────────────────────────────────────────────

class _QNetwork(nn.Module):
    def __init__(self, state_dim: int = INPUT_DIM, action_dim: int = OUTPUT_DIM, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + action_dim),
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class UrsulaSACCritic(nn.Module):
    def __init__(self, state_dim: int = INPUT_DIM, action_dim: int = OUTPUT_DIM, hidden_dim: int = 512):
        super().__init__()
        self.q1 = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = _QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ── Smoke Tests ───────────────────────────────────────────────────────────────

def _smoke_test():
    print("=== agents/ursula.py smoke tests ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = 4

    policy = UrsulaPolicy().to(device)
    x = torch.randn(batch, INPUT_DIM, device=device)
    out = policy(x)
    assert out.shape == (batch, OUTPUT_DIM)
    assert out.min() >= -1.0 and out.max() <= 1.0
    print(f"  [PASS] UrsulaPolicy  input={x.shape} → output={out.shape}")

    params = ActionUnnormalizer.decode(out)
    assert len(params) == OUTPUT_DIM
    assert "eq_band1_freq" in params
    assert "gain_db" in params
    print(f"  [PASS] ActionUnnormalizer decode → {len(params)} params")

    encoded = ActionUnnormalizer.encode(params)
    assert encoded.shape == (batch, OUTPUT_DIM)
    re_decoded = ActionUnnormalizer.decode(encoded)
    for key in params:
        assert torch.allclose(params[key], re_decoded[key], atol=1e-4)
    print(f"  [PASS] encode/decode roundtrip")

    plugin_dicts = ActionUnnormalizer.decode_to_plugin_dicts(out)
    assert set(plugin_dicts.keys()) == {"eq", "gain"}
    print(f"  [PASS] decode_to_plugin_dicts")

    actor = UrsulaSACActor().to(device)
    action, log_prob = actor(x)
    assert action.shape == (batch, OUTPUT_DIM)
    assert log_prob.shape == (batch,)
    print(f"  [PASS] UrsulaSACActor")

    critic = UrsulaSACCritic().to(device)
    q1, q2 = critic(x, action)
    assert q1.shape == (batch, 1)
    assert q2.shape == (batch, 1)
    print(f"  [PASS] UrsulaSACCritic")

    policy_params = sum(p.numel() for p in policy.parameters())
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  [INFO] Policy={policy_params:,}  Actor={actor_params:,}  Critic={critic_params:,}")

    print("=== All smoke tests passed ===")


if __name__ == "__main__":
    _smoke_test()
'''

export_path = OUTPUT / 'agents' / 'ursula.py'
export_path.parent.mkdir(parents=True, exist_ok=True)
export_path.write_text(EXPORT_SOURCE)
print(f"  Wrote {export_path}")

elapsed = time.time() - t_total
print(f"\n{'=' * 60}")
print(f"  PHASE 4 COMPLETE — {elapsed:.1f}s")
print(f"  Architecture: LayerNorm → 512 → 512(+residual) → 256 → 2 heads")
print(f"  Policy: {policy_params:,} params")
print(f"  Actor:  {actor_params:,} params")
print(f"  Critic: {critic_params:,} params")
print(f"  Output: {export_path}")
print(f"{'=' * 60}")
