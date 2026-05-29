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
        hidden_dim: int = 512,
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
        self.target_entropy = -float(action_dim)  # heuristic: -dim(A)
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
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
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

            actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ── Alpha auto-tune ──
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

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
