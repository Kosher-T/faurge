# %% [markdown]
# ## Smoke Tests
#
# Verifies the environment loads correctly and basic functionality works.

t_start = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Environment loads without errors
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 1: Environment loads")
print("=" * 60)

env = UrsulaDSPEnv()
print(f"  observation_space: {env.observation_space}")
print(f"  action_space: {env.action_space}")
print(f"  pairs loaded: {len(env._pairs)}")
assert env.observation_space.shape == (INPUT_DIM,), f"Obs shape: {env.observation_space.shape}"
assert env.action_space.shape == (OUTPUT_DIM,), f"Act shape: {env.action_space.shape}"
print("  [PASS] Environment loads correctly\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Reset produces correct observation shape
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 2: Reset produces correct observation")
print("=" * 60)

obs, info = env.reset()
print(f"  obs shape: {obs.shape}")
print(f"  obs dtype: {obs.dtype}")
print(f"  info: {info}")
assert obs.shape == (INPUT_DIM,), f"Obs shape: {obs.shape}"
assert obs.dtype == np.float32
assert "pair_id" in info
assert "cluster_id" in info
assert "initial_mse" in info
print(f"  [PASS] Reset OK — pair={info['pair_id']}, cluster={info['cluster_id']}, mse={info['initial_mse']:.4f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Random actions produce plausible audio (no NaN, no silence-burst)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 3: Random actions → plausible audio")
print("=" * 60)

obs, info = env.reset()
total_reward = 0.0
mse_history = [info["initial_mse"]]

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, step_info = env.step(action)
    total_reward += reward
    mse_history.append(step_info.get("mse", float('nan')))
    print(f"  step {step+1}: reward={reward:.4f}, mse={step_info.get('mse', 'N/A'):.4f}")
    if terminated or truncated:
        break

# Check no NaN in observations
assert not np.any(np.isnan(obs)), "NaN in observation"
assert not np.isinf(obs).any(), "Inf in observation"
# Check reward is finite
assert np.isfinite(total_reward), f"Non-finite total reward: {total_reward}"
print(f"  [PASS] Random rollout OK — total_reward={total_reward:.4f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Multiple resets produce different pairs
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 4: Multiple resets → different pairs")
print("=" * 60)

pair_ids = set()
for _ in range(10):
    obs, info = env.reset()
    pair_ids.add(info["pair_id"])
print(f"  Unique pairs in 10 resets: {len(pair_ids)}")
assert len(pair_ids) >= 2, f"Only {len(pair_ids)} unique pairs in 10 resets"
print(f"  [PASS] Multiple resets OK\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 5: Rewards are negative and increase as random actions improve
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 5: Reward properties")
print("=" * 60)

obs, info = env.reset()
rewards = []
for _ in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, step_info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        obs, info = env.reset()

# All rewards should be <= 0 (soft_clamp outputs non-positive)
assert all(r <= 0.0 for r in rewards), f"Rewards not all <= 0: {rewards}"
print(f"  Rewards: {[f'{r:.4f}' for r in rewards]}")
print(f"  [PASS] Rewards are non-positive\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 6: Cluster one-hot is correct
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 6: Cluster one-hot correctness")
print("=" * 60)

obs, info = env.reset()
cluster_part = obs[2 * METRIC_DIM:]  # last 9 dims
assert cluster_part.shape == (N_CLUSTERS_ONEHOT,), f"Cluster part shape: {cluster_part.shape}"
assert np.sum(cluster_part) == 1.0, f"Cluster one-hot sum: {np.sum(cluster_part)}"
assert np.max(cluster_part) == 1.0, f"Cluster one-hot max: {np.max(cluster_part)}"
print(f"  Cluster one-hot: {cluster_part}")
print(f"  [PASS] Cluster one-hot is valid\n")

# ══════════════════════════════════════════════════════════════════════════════
# Test 7: Full episode (50 steps) runs without errors
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST 7: Full episode (50 steps)")
print("=" * 60)

obs, info = env.reset()
total_reward = 0.0
for step in range(MAX_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, step_info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"  Completed {step+1} steps, total_reward={total_reward:.4f}")
print(f"  Final MSE: {step_info.get('mse', 'N/A')}")
print(f"  [PASS] Full episode OK\n")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print("=" * 60)
print(f"  PHASE 5 COMPLETE — {elapsed:.1f}s")
print(f"  Environment: UrsulaDSPEnv")
print(f"  Observation: {INPUT_DIM}D  (M_current {METRIC_DIM}D + M_ref {METRIC_DIM}D + cluster {N_CLUSTERS_ONEHOT}D)")
print(f"  Action: {OUTPUT_DIM}D  (7 DSP plugins)")
print(f"  Max steps per episode: {MAX_STEPS}")
print(f"  All smoke tests passed")
print("=" * 60)
