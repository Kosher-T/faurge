# %% [markdown]
# ## Replay Buffer & SAC Agent
#
# Standard SAC components: circular replay buffer, twin Q-networks with
# clipped double-Q, auto-tuned entropy coefficient, and soft target updates.

# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy RL."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self._size = 0
        self._pos = 0

        self.states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, 1, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, 1, dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.states[self._pos] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self._pos] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self._pos] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self._pos] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self._pos] = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self._size


# ══════════════════════════════════════════════════════════════════════════════
# SAC Agent
# ══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    """
    Soft Actor-Critic agent for Ursula DSP policy training.

    Components:
        - UrsulaSACActor (policy + learned log-std)
        - UrsulaSACCritic (twin Q-networks)
        - Target networks (hard copies, soft-updated)
        - Auto-tuned entropy coefficient alpha
    """

    def __init__(
        self,
        state_dim: int = INPUT_DIM,
        action_dim: int = OUTPUT_DIM,
        hidden_dim: int = 944,
        actor_lr: float = ACTOR_LR,
        critic_lr: float = CRITIC_LR,
        alpha_lr: float = ALPHA_LR,
        gamma: float = GAMMA,
        tau: float = TAU,
        device: str = DEVICE,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim

        # Actor
        self.actor = UrsulaSACActor(
            input_dim=state_dim, output_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics
        self.critic = UrsulaSACCritic(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Target networks (initialized as copies)
        self.critic_target = UrsulaSACCritic(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Auto-tuned entropy coefficient alpha
        self.target_entropy = -float(action_dim) / 4.0  # less aggressive: ~-57 instead of -113
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Counters
        self._update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for a single state (no batch dim)."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor(s, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch) -> Dict[str, float]:
        """
        Run one SAC update step on a batch from the replay buffer.

        Returns dict of training metrics: critic_loss, actor_loss, alpha, alpha_loss.
        """
        states, actions, rewards, next_states, dones = batch

        # ── Critic update ──
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs.unsqueeze(-1)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update (delayed) ──
        actor_loss_val = 0.0
        alpha_loss_val = 0.0
        self._update_count += 1

        if self._update_count % POLICY_DELAY == 0:
            new_actions, log_probs = self.actor(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            # L2 regularization toward zero action (identity prior)
            # Light regularization — pretrained policy is already near good actions
            action_reg = 0.0001 * (new_actions ** 2).mean()
            actor_loss = (self.alpha.detach() * log_probs.unsqueeze(-1) - q_new).mean() + action_reg

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ── Alpha auto-tune with floor ──
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Enforce alpha minimum (prevent entropy death)
            # Use clamp on log_alpha directly — more robust than conditional assignment
            with torch.no_grad():
                min_log_alpha = torch.tensor([np.log(ALPHA_MIN)], device=self.device)
                self.log_alpha.data = torch.maximum(self.log_alpha.data, min_log_alpha)

            actor_loss_val = actor_loss.item()
            alpha_loss_val = alpha_loss.item()

            # ── Soft update target networks ──
            self._soft_update_target()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss_val,
        }

    def _soft_update_target(self):
        """Polyak averaging: θ_target ← τ·θ + (1−τ)·θ_target"""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def load_pretrained_policy(self, path: Path):
        """Load pretrained UrsulaPolicy weights as warm-start for the actor.

        Maps from UrsulaPolicy (plugin_heads.* → mu_heads.*) and loads
        the shared trunk. log_std_heads and critic networks remain fresh.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        policy_sd = ckpt["policy_state_dict"]

        # Separate trunk keys vs plugin_head keys
        trunk_keys = {k: v for k, v in policy_sd.items() if k.startswith("trunk_")}
        plugin_keys = {k: v for k, v in policy_sd.items() if k.startswith("plugin_heads.")}

        # Load trunk (same key names in UrsulaPolicy and UrsulaSACActor)
        missing, unexpected = self.actor.load_state_dict(trunk_keys, strict=False)
        if missing:
            print(f"  [PRETRAIN] Trunk missing keys (expected — log_std_heads): {len(missing)}")
        if unexpected:
            print(f"  [PRETRAIN] Trunk unexpected keys: {unexpected}")

        # Map plugin_heads.* → mu_heads.*
        mu_sd = self.actor.mu_heads.state_dict()
        for k, v in plugin_keys.items():
            # k = "plugin_heads.eq.weight" → "eq.weight"
            head_key = k[len("plugin_heads."):]
            if head_key in mu_sd:
                mu_sd[head_key] = v
        self.actor.mu_heads.load_state_dict(mu_sd)

        n_trunk = len(trunk_keys)
        n_plugin = len(plugin_keys)
        total = sum(p.numel() for p in self.actor.parameters())
        loaded = sum(v.numel() for v in trunk_keys.values()) + sum(v.numel() for v in plugin_keys.values())
        print(f"  [PRETRAIN] Loaded {n_trunk} trunk + {n_plugin} plugin keys ({loaded:,}/{total:,} params)")
        print(f"  [PRETRAIN] log_std_heads and critics initialized fresh")

    def save_checkpoint(self, path: Path, step: int, extra: dict = None):
        """Save full agent state to disk."""
        ckpt = {
            "step": step,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self._update_count,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path):
        """Restore agent state from disk."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        self.critic_target.load_state_dict(ckpt["critic_target_state"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self._update_count = ckpt.get("update_count", 0)
        return ckpt.get("step", 0)


print("ReplayBuffer and SACAgent defined")
