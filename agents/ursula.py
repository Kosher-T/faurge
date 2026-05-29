"""
agents/ursula.py — Ursula Policy Network & SAC Wrappers
========================================================

Pure PyTorch implementation of Ursula's DSP policy network.
No Kaggle dependency — runs entirely locally.

Architecture
------------
Input:  143D  (M_degraded 67D || M_reference 67D || cluster_onehot 9D)
Output: 227D  (tanh-activated, scaled to each plugin parameter's real range)

Hidden: LayerNorm → 512 → ReLU → 512 → ReLU → 256 → ReLU → 227

The 227D output maps to 7 DSP plugins in cascade order:
  1. EQ         (31 bands × 6 params = 186D)
  2. Compressor (14D)
  3. Esser      (6D)
  4. Saturator  (7D)
  5. Limiter    (6D)
  6. Transient  (6D)
  7. Gain       (2D)

Usage
-----
    import torch
    from agents.ursula import UrsulaPolicy, ActionUnnormalizer

    policy = UrsulaPolicy()
    x = torch.randn(1, 143)
    raw_out = policy(x)          # (1, 227) in [-1, 1]
    params = ActionUnnormalizer.decode(raw_out)  # dict of plugin params
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM = 143       # 67 + 67 + 9
OUTPUT_DIM = 227      # all plugin params flattened
N_CLUSTERS = 8        # K voice clusters
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1  # +1 for "unknown"
METRIC_DIM = 67       # LTAS 64 + LUFS 1 + Dyn 2


# ── Parameter Specification ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamRange:
    """Single parameter: name, low/high bounds, log-scale flag."""
    name: str
    low: float
    high: float
    log: bool = False


# EQ: 31 bands × 6 params = 186D
EQ_PARAM_RANGES: List[ParamRange] = []
for _b in range(31):
    EQ_PARAM_RANGES.extend([
        ParamRange(f"eq_band{_b+1}_freq",       20.0,     20_000.0, log=True),
        ParamRange(f"eq_band{_b+1}_gain",       -24.0,        24.0),
        ParamRange(f"eq_band{_b+1}_q",            0.1,         10.0),
        ParamRange(f"eq_band{_b+1}_filter_type",  0.0,          6.0),   # categorical 0–6
        ParamRange(f"eq_band{_b+1}_stereo_skew", -6.0,          6.0),
        ParamRange(f"eq_band{_b+1}_dynamic_depth", 0.0,          1.0),
    ])

# Compressor: 14D
COMP_PARAM_RANGES = [
    ParamRange("comp_threshold",      -60.0,     0.0),
    ParamRange("comp_ratio",           1.0,     20.0),
    ParamRange("comp_attack",          0.1,    100.0),
    ParamRange("comp_release",        10.0,   1000.0),
    ParamRange("comp_knee",            0.0,     12.0),
    ParamRange("comp_lookahead",       0.0,     10.0),
    ParamRange("comp_hold",            0.0,    200.0),
    ParamRange("comp_wet_dry",         0.0,      1.0),
    ParamRange("comp_stereo_link",     0.0,      1.0),
    ParamRange("comp_sidechain_hp",   20.0,    500.0),
    ParamRange("comp_sidechain_lp",  500.0,  20_000.0, log=True),
    ParamRange("comp_saturate_drive",  0.0,     12.0),
    ParamRange("comp_output_trim",   -12.0,     12.0),
    ParamRange("comp_detector_type",   0.0,      3.0),   # categorical 0–3
]

# Esser: 6D
ESSER_PARAM_RANGES = [
    ParamRange("esser_center",       4000.0,  10_000.0, log=True),
    ParamRange("esser_threshold",    -60.0,      0.0),
    ParamRange("esser_ratio",          0.25,    20.0),
    ParamRange("esser_bandwidth",    500.0,   4000.0, log=True),
    ParamRange("esser_attack",         0.1,     50.0),
    ParamRange("esser_release",       10.0,    500.0),
]

# Saturator: 7D
SAT_PARAM_RANGES = [
    ParamRange("sat_drive",           0.0,     24.0),
    ParamRange("sat_mix",             0.0,      1.0),
    ParamRange("sat_type",            0.0,      3.0),   # categorical 0–3
    ParamRange("sat_hpf",            20.0,    500.0),
    ParamRange("sat_lpf",          2000.0,  20_000.0, log=True),
    ParamRange("sat_oversampling",    0.0,      3.0),   # categorical 0–3
    ParamRange("sat_output_trim",   -12.0,     12.0),
]

# Limiter: 6D
LIM_PARAM_RANGES = [
    ParamRange("lim_ceiling",       -12.0,      0.0),
    ParamRange("lim_release",         1.0,    500.0),
    ParamRange("lim_lookahead",       0.0,     10.0),
    ParamRange("lim_clip_mode",       0.0,      1.0),   # categorical 0–1
    ParamRange("lim_stereo_link",     0.0,      1.0),
    ParamRange("lim_oversampling",    0.0,      3.0),   # categorical 0–3
]

# Transient: 6D
TRANS_PARAM_RANGES = [
    ParamRange("trans_attack_gain",  -24.0,     24.0),
    ParamRange("trans_sustain_gain", -24.0,     24.0),
    ParamRange("trans_attack_time",    0.1,     50.0),
    ParamRange("trans_release_time",  10.0,    500.0),
    ParamRange("trans_sensitivity",  -30.0,      0.0),
    ParamRange("trans_mix",            0.0,      1.0),
]

# Gain: 2D
GAIN_PARAM_RANGES = [
    ParamRange("gain_db",           -12.0,     12.0),
    ParamRange("stereo_balance",     -1.0,      1.0),
]

# Master list: all 227D in order
ALL_PARAM_RANGES: List[ParamRange] = (
    EQ_PARAM_RANGES
    + COMP_PARAM_RANGES
    + ESSER_PARAM_RANGES
    + SAT_PARAM_RANGES
    + LIM_PARAM_RANGES
    + TRANS_PARAM_RANGES
    + GAIN_PARAM_RANGES
)

assert len(ALL_PARAM_RANGES) == OUTPUT_DIM, (
    f"Param count mismatch: {len(ALL_PARAM_RANGES)} != {OUTPUT_DIM}"
)


# ── Slice indices for each plugin ─────────────────────────────────────────────

PLUGIN_SLICES: Dict[str, Tuple[int, int]] = {}
_offset = 0
for _name, _count in [
    ("eq",        31 * 6),   # 186
    ("compressor",      14),
    ("esser",            6),
    ("saturator",        7),
    ("limiter",          6),
    ("transient",        6),
    ("gain",             2),
]:
    PLUGIN_SLICES[_name] = (_offset, _offset + _count)
    _offset += _count


# ── Categorical parameter indices (within the 227D output) ────────────────────
# These dimensions use softmax at inference (argmax) instead of tanh scaling.

CATEGORICAL_INDICES: Dict[str, List[int]] = {
    "eq_filter_type": list(range(2, 186, 6)),      # every 6th starting at index 2
    "comp_detector_type": [186 + 13],               # index 199
    "sat_type": [206 + 2],                          # index 208
    "sat_oversampling": [206 + 5],                  # index 211
    "lim_clip_mode": [213 + 3],                     # index 216
    "lim_oversampling": [213 + 5],                  # index 218
}


# ── Policy Network ────────────────────────────────────────────────────────────

class UrsulaPolicy(nn.Module):
    """
    Ursula's feed-forward policy network.

    Input:  (batch, 143) — [M_degraded(67), M_reference(67), cluster_onehot(9)]
    Output: (batch, 227) — tanh-activated raw action in [-1, 1]

    The output is *not* scaled to real parameter ranges here.
    Use ActionUnnormalizer.decode() to convert to plugin params.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        n_clusters: int = N_CLUSTERS_ONEHOT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),   # 256
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh(),                                 # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 143) — concatenated metrics + cluster one-hot
        Returns:
            (batch, 227) — raw action in [-1, 1]
        """
        return self.net(x)


# ── Action Unnormalizer ───────────────────────────────────────────────────────

class ActionUnnormalizer:
    """
    Converts tanh output [-1, 1] → real plugin parameter values.

    For continuous params: linear interpolation [low, high].
    For log-scaled params: log-space interpolation.
    For categorical params: argmax over discretized bins.

    Static methods so it can be used without instantiation (stateless).
    """

    @staticmethod
    def decode(
        raw: torch.Tensor,
        param_ranges: List[ParamRange] | None = None,
        categorical_indices: Dict[str, List[int]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode raw tanh output to a dict of plugin parameters.

        Args:
            raw: (batch, 227) tensor in [-1, 1]
            param_ranges: list of ParamRange (defaults to ALL_PARAM_RANGES)
            categorical_indices: dict mapping group name → index list

        Returns:
            dict mapping param_name → (batch,) tensor of real values
        """
        if param_ranges is None:
            param_ranges = ALL_PARAM_RANGES
        if categorical_indices is None:
            categorical_indices = CATEGORICAL_INDICES

        # Flatten categorical_indices into a set for fast lookup
        cat_set: set = set()
        for indices in categorical_indices.values():
            cat_set.update(indices)

        result: Dict[str, torch.Tensor] = {}
        for i, pr in enumerate(param_ranges):
            val = raw[:, i]  # (batch,) in [-1, 1]

            if i in cat_set:
                # Categorical: map to nearest integer bin
                result[pr.name] = torch.round(
                    (val + 1.0) * 0.5 * (pr.high - pr.low) + pr.low
                ).clamp(pr.low, pr.high)
            elif pr.log:
                # Log-scale: map [-1,1] → [log(low), log(high)] → exp
                log_low = math.log(max(pr.low, 1e-8))
                log_high = math.log(max(pr.high, 1e-8))
                result[pr.name] = torch.exp(
                    (val + 1.0) * 0.5 * (log_high - log_low) + log_low
                )
            else:
                # Linear: map [-1,1] → [low, high]
                result[pr.name] = (val + 1.0) * 0.5 * (pr.high - pr.low) + pr.low

        return result

    @staticmethod
    def decode_to_plugin_dicts(
        raw: torch.Tensor,
    ) -> Dict[str, dict]:
        """
        Decode raw output and group into per-plugin config dicts
        matching the portable plugin API signatures.

        Returns:
            dict with keys: eq, compressor, esser, saturator, limiter, transient, gain
            Each value is a dict of param_name → value (scalar or list for EQ bands).
        """
        params = ActionUnnormalizer.decode(raw)
        batch_size = raw.shape[0]

        # ── EQ: group 31 bands ──
        eq_bands: List[dict] = []
        for b in range(31):
            base = b * 6
            freq = params[f"eq_band{b+1}_freq"]
            gain = params[f"eq_band{b+1}_gain"]
            q = params[f"eq_band{b+1}_q"]
            ftype_idx = params[f"eq_band{b+1}_filter_type"]
            skew = params[f"eq_band{b+1}_stereo_skew"]
            dyn = params[f"eq_band{b+1}_dynamic_depth"]

            # Map float filter_type index → string
            _FTYPES = ["peak", "low_shelf", "high_shelf", "highpass", "lowpass", "bandpass", "notch"]
            ftype = _FTYPES[int(ftype_idx.round().clamp(0, 6).item())] if batch_size == 1 else _FTYPES

            eq_bands.append({
                "freq_hz": freq.item() if batch_size == 1 else freq,
                "gain_db": gain.item() if batch_size == 1 else gain,
                "q": q.item() if batch_size == 1 else q,
                "filter_type": ftype,
                "stereo_skew_db": skew.item() if batch_size == 1 else skew,
                "dynamic_depth": dyn.item() if batch_size == 1 else dyn,
            })

        # ── Compressor ──
        _DETECT_TYPES = ["RMS", "peak", "feed_forward", "feed_back"]
        comp_det = params["comp_detector_type"]
        comp = {
            "threshold_db": params["comp_threshold"].item() if batch_size == 1 else params["comp_threshold"],
            "ratio": params["comp_ratio"].item() if batch_size == 1 else params["comp_ratio"],
            "attack_ms": params["comp_attack"].item() if batch_size == 1 else params["comp_attack"],
            "release_ms": params["comp_release"].item() if batch_size == 1 else params["comp_release"],
            "knee_db": params["comp_knee"].item() if batch_size == 1 else params["comp_knee"],
            "lookahead_ms": params["comp_lookahead"].item() if batch_size == 1 else params["comp_lookahead"],
            "hold_ms": params["comp_hold"].item() if batch_size == 1 else params["comp_hold"],
            "wet_dry_mix": params["comp_wet_dry"].item() if batch_size == 1 else params["comp_wet_dry"],
            "stereo_link": params["comp_stereo_link"].item() if batch_size == 1 else params["comp_stereo_link"],
            "sidechain_hp_hz": params["comp_sidechain_hp"].item() if batch_size == 1 else params["comp_sidechain_hp"],
            "sidechain_lp_hz": params["comp_sidechain_lp"].item() if batch_size == 1 else params["comp_sidechain_lp"],
            "saturate_drive_db": params["comp_saturate_drive"].item() if batch_size == 1 else params["comp_saturate_drive"],
            "output_trim_db": params["comp_output_trim"].item() if batch_size == 1 else params["comp_output_trim"],
            "detector_type": _DETECT_TYPES[int(comp_det.round().clamp(0, 3).item())] if batch_size == 1 else _DETECT_TYPES,
        }

        # ── Esser ──
        esser = {
            "center_freq_hz": params["esser_center"].item() if batch_size == 1 else params["esser_center"],
            "threshold_db": params["esser_threshold"].item() if batch_size == 1 else params["esser_threshold"],
            "ratio": params["esser_ratio"].item() if batch_size == 1 else params["esser_ratio"],
            "bandwidth_hz": params["esser_bandwidth"].item() if batch_size == 1 else params["esser_bandwidth"],
            "attack_ms": params["esser_attack"].item() if batch_size == 1 else params["esser_attack"],
            "release_ms": params["esser_release"].item() if batch_size == 1 else params["esser_release"],
        }

        # ── Saturator ──
        _SAT_TYPES = ["tube", "tape", "diode", "asymmetric"]
        _OS_TYPES = [1, 2, 4, 8]
        sat_type = params["sat_type"]
        sat_os = params["sat_oversampling"]
        sat = {
            "drive_db": params["sat_drive"].item() if batch_size == 1 else params["sat_drive"],
            "mix": params["sat_mix"].item() if batch_size == 1 else params["sat_mix"],
            "sat_type": _SAT_TYPES[int(sat_type.round().clamp(0, 3).item())] if batch_size == 1 else _SAT_TYPES,
            "hpf_hz": params["sat_hpf"].item() if batch_size == 1 else params["sat_hpf"],
            "lpf_hz": params["sat_lpf"].item() if batch_size == 1 else params["sat_lpf"],
            "oversampling": _OS_TYPES[int(sat_os.round().clamp(0, 3).item())] if batch_size == 1 else _OS_TYPES,
            "output_trim_db": params["sat_output_trim"].item() if batch_size == 1 else params["sat_output_trim"],
        }

        # ── Limiter ──
        _CLIP_MODES = ["hard", "soft"]
        lim_clip = params["lim_clip_mode"]
        lim_os = params["lim_oversampling"]
        lim = {
            "ceiling_db": params["lim_ceiling"].item() if batch_size == 1 else params["lim_ceiling"],
            "release_ms": params["lim_release"].item() if batch_size == 1 else params["lim_release"],
            "lookahead_ms": params["lim_lookahead"].item() if batch_size == 1 else params["lim_lookahead"],
            "clip_mode": _CLIP_MODES[int(lim_clip.round().clamp(0, 1).item())] if batch_size == 1 else _CLIP_MODES,
            "stereo_link": params["lim_stereo_link"].item() if batch_size == 1 else params["lim_stereo_link"],
            "oversampling": _OS_TYPES[int(lim_os.round().clamp(0, 3).item())] if batch_size == 1 else _OS_TYPES,
        }

        # ── Transient ──
        trans = {
            "attack_gain_db": params["trans_attack_gain"].item() if batch_size == 1 else params["trans_attack_gain"],
            "sustain_gain_db": params["trans_sustain_gain"].item() if batch_size == 1 else params["trans_sustain_gain"],
            "attack_time_ms": params["trans_attack_time"].item() if batch_size == 1 else params["trans_attack_time"],
            "release_time_ms": params["trans_release_time"].item() if batch_size == 1 else params["trans_release_time"],
            "sensitivity_db": params["trans_sensitivity"].item() if batch_size == 1 else params["trans_sensitivity"],
            "mix": params["trans_mix"].item() if batch_size == 1 else params["trans_mix"],
        }

        # ── Gain ──
        g = {
            "gain_db": params["gain_db"].item() if batch_size == 1 else params["gain_db"],
            "stereo_balance": params["stereo_balance"].item() if batch_size == 1 else params["stereo_balance"],
        }

        return {
            "eq": eq_bands,
            "compressor": comp,
            "esser": esser,
            "saturator": sat,
            "limiter": lim,
            "transient": trans,
            "gain": g,
        }


# ── SAC Actor ─────────────────────────────────────────────────────────────────

class UrsulaSACActor(nn.Module):
    """
    SAC actor: wraps the deterministic policy with learned log-std
    for Gaussian exploration.

    During training, samples from N(mu, sigma).
    During eval, returns tanh(mu).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 143)
            deterministic: if True, skip sampling
        Returns:
            action: (batch, 227) in [-1, 1]
            log_prob: (batch,) log-probability of the action
        """
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mu)
            # log_prob under deterministic is not meaningful; return 0
            return action, torch.zeros(x.shape[0], device=x.device)

        # Reparameterization trick
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # reparameterized
        action = torch.tanh(z)

        # Correct for tanh squashing (change of variables)
        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Convenience: return only the action tensor."""
        action, _ = self.forward(x, deterministic=deterministic)
        return action


# ── SAC Critic (Twin Q-Networks) ─────────────────────────────────────────────

class _QNetwork(nn.Module):
    """Single Q-network: (state, action) → Q-value."""

    def __init__(
        self,
        state_dim: int = INPUT_DIM,
        action_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
    ):
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
        """Returns Q-value (batch, 1)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class UrsulaSACCritic(nn.Module):
    """
    Twin Q-networks for SAC (clipped double-Q).

    Q1(state, action) and Q2(state, action) are independent.
    Returns (q1, q2).
    """

    def __init__(
        self,
        state_dim: int = INPUT_DIM,
        action_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.q1 = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = _QNetwork(state_dim, action_dim, hidden_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return min(Q1, Q2) — used for target network updates."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ── Smoke Tests ───────────────────────────────────────────────────────────────

def _smoke_test():
    """Run basic shape and value checks. Import-guarded."""
    print("=== agents/ursula.py smoke tests ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = 4

    # --- UrsulaPolicy ---
    policy = UrsulaPolicy().to(device)
    x = torch.randn(batch, INPUT_DIM, device=device)
    out = policy(x)
    assert out.shape == (batch, OUTPUT_DIM), f"Policy shape: {out.shape}"
    assert out.min() >= -1.0 and out.max() <= 1.0, "Policy output out of [-1, 1]"
    print(f"  [PASS] UrsulaPolicy  input={x.shape} → output={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

    # --- ActionUnnormalizer ---
    params = ActionUnnormalizer.decode(out)
    assert len(params) == OUTPUT_DIM, f"Unnormalized params count: {len(params)}"
    # Check a few specific params
    assert "eq_band1_freq" in params
    assert "comp_threshold" in params
    assert "gain_db" in params
    freq_val = params["eq_band1_freq"]
    assert (freq_val >= 20.0).all() and (freq_val <= 20_000.0).all(), f"EQ freq out of range: {freq_val}"
    print(f"  [PASS] ActionUnnormalizer decode → {len(params)} params")

    # --- decode_to_plugin_dicts ---
    plugin_dicts = ActionUnnormalizer.decode_to_plugin_dicts(out)
    assert set(plugin_dicts.keys()) == {"eq", "compressor", "esser", "saturator", "limiter", "transient", "gain"}
    assert len(plugin_dicts["eq"]) == 31
    assert "threshold_db" in plugin_dicts["compressor"]
    print(f"  [PASS] decode_to_plugin_dicts → keys={list(plugin_dicts.keys())}")

    # --- UrsulaSACActor ---
    actor = UrsulaSACActor().to(device)
    action, log_prob = actor(x)
    assert action.shape == (batch, OUTPUT_DIM)
    assert log_prob.shape == (batch,)
    assert action.min() >= -1.0 and action.max() <= 1.0
    print(f"  [PASS] UrsulaSACActor  action={action.shape}, log_prob={log_prob.shape}")

    # Deterministic mode
    det_action, det_lp = actor(x, deterministic=True)
    assert (det_lp == 0).all()
    print(f"  [PASS] UrsulaSACActor deterministic mode")

    # --- UrsulaSACCritic ---
    critic = UrsulaSACCritic().to(device)
    q1, q2 = critic(x, action)
    assert q1.shape == (batch, 1)
    assert q2.shape == (batch, 1)
    qmin = critic.q_min(x, action)
    assert qmin.shape == (batch, 1)
    print(f"  [PASS] UrsulaSACCritic  Q1={q1.shape}, Q2={q2.shape}, Q_min={qmin.shape}")

    # --- Parameter count ---
    policy_params = sum(p.numel() for p in policy.parameters())
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  [INFO] Param counts: Policy={policy_params:,}  Actor={actor_params:,}  Critic={critic_params:,}")

    # --- Identity test: identical metrics → near-identity output ---
    policy_cpu = UrsulaPolicy()
    policy_cpu.eval()
    with torch.no_grad():
        # Build input where M_degraded == M_reference
        metrics = torch.randn(1, METRIC_DIM)
        ident_input = torch.cat([metrics, metrics, torch.zeros(1, N_CLUSTERS_ONEHOT)], dim=-1)
        ident_out = policy_cpu(ident_input)
    # gain_db should be near 0 (centered at 0 in [-12, +12])
    gain_idx = OUTPUT_DIM - 2  # second-to-last is gain_db
    gain_raw = ident_out[0, gain_idx].item()
    gain_real = (gain_raw + 1.0) * 0.5 * (12.0 - (-12.0)) + (-12.0)
    assert abs(gain_real) < 3.0, f"Identity test: gain_db={gain_real:.2f} (expected near 0)"
    print(f"  [PASS] Identity test: identical metrics → gain_db={gain_real:.2f} dB (near 0)")

    print("=== All smoke tests passed ===\n")


if __name__ == "__main__":
    _smoke_test()
