# %% [markdown]
# ## Architecture Definition — The Core
#
# Three `nn.Module` classes:
# - **UrsulaPolicy**: Deterministic feed-forward with per-plugin output heads
# - **UrsulaSACActor**: Stochastic policy for SAC (reparameterization trick)
# - **UrsulaSACCritic**: Twin Q-networks (clipped double-Q)

# ══════════════════════════════════════════════════════════════════════════════
# UrsulaPolicy — Deterministic Policy Network
# ══════════════════════════════════════════════════════════════════════════════

class UrsulaPolicy(nn.Module):
    """
    Ursula's feed-forward policy network with per-plugin output heads.

    Input:  (batch, 143) — [M_degraded(67), M_reference(67), cluster_onehot(9)]
    Output: (batch, 125) — tanh-activated raw action in [-1, 1]

    Trunk:
        LayerNorm(143) → Linear(143, 512) → ReLU → Dropout
        Linear(512, 512) → ReLU → Dropout + Residual Skip
        Linear(512, 256) → ReLU

    Output heads: 2 independent Linear(256, plugin_dim) → Tanh
    """

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
        """Initialize gain head bias so untrained output ≈ 0 dB gain."""
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


# ══════════════════════════════════════════════════════════════════════════════
# ActionUnnormalizer
# ══════════════════════════════════════════════════════════════════════════════

class ActionUnnormalizer:
    """
    Converts tanh output [-1, 1] ↔ real plugin parameter values.
    Static methods — stateless, no instantiation needed.
    """

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

        # Continuous non-log: linear scale
        cont_mask = ~cat_mask & ~is_log
        vals[:, cont_mask] = (
            (raw[:, cont_mask] + 1.0) * 0.5
            * (highs[cont_mask] - lows[cont_mask]) + lows[cont_mask]
        )

        # Continuous log: log-space scale
        log_mask = ~cat_mask & is_log
        log_lows = torch.log(lows[log_mask].clamp(min=1e-8))
        log_highs = torch.log(highs[log_mask].clamp(min=1e-8))
        vals[:, log_mask] = torch.exp(
            (raw[:, log_mask] + 1.0) * 0.5 * (log_highs - log_lows) + log_lows
        )

        # Categorical: nearest integer bin
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
        """Encode real parameter values to tanh output [-1, 1]."""
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
        """Decode raw output and group into per-plugin config dicts."""
        params = ActionUnnormalizer.decode(raw)
        batch_size = raw.shape[0]

        # EQ bands
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

        # Gain
        g = {
            "gain_db": params["gain_db"].item() if batch_size == 1 else params["gain_db"],
        }

        return {
            "eq": eq_bands, "gain": g,
        }


# ══════════════════════════════════════════════════════════════════════════════
# UrsulaSACActor — Stochastic Policy for SAC
# ══════════════════════════════════════════════════════════════════════════════

class UrsulaSACActor(nn.Module):
    """
    SAC actor: shared trunk with per-plugin mu and log_std heads.
    Reparameterization trick with squashed Gaussian.
    """

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


# ══════════════════════════════════════════════════════════════════════════════
# UrsulaSACCritic — Twin Q-Networks
# ══════════════════════════════════════════════════════════════════════════════

class _QNetwork(nn.Module):
    """Single Q-network: (state, action) → Q-value."""

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
    """Twin Q-networks for SAC (clipped double-Q)."""

    def __init__(self, state_dim: int = INPUT_DIM, action_dim: int = OUTPUT_DIM, hidden_dim: int = 512):
        super().__init__()
        self.q1 = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = _QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

print("Architecture classes defined: UrsulaPolicy, ActionUnnormalizer, UrsulaSACActor, UrsulaSACCritic")
