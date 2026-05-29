# %% [markdown]
# ## Execute Training
#
# Runs the full SAC training loop with curriculum expansion, cluster masking,
# periodic logging, checkpointing, and audio rollout recording.

import soundfile as sf

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Environment & Agent
# ══════════════════════════════════════════════════════════════════════════════

env = UrsulaDSPEnv(mode="train", max_pairs=CURRICULUM[0]["max_pairs"])
agent = SACAgent()
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, INPUT_DIM, OUTPUT_DIM, device=DEVICE)

print(f"Environment: {len(env._pairs)} pairs (Phase A: {CURRICULUM[0]['max_pairs']})")
print(f"Agent: {sum(p.numel() for p in agent.actor.parameters()):,} actor params, "
      f"{sum(p.numel() for p in agent.critic.parameters()):,} critic params")

# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_STEPS = 100_000
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 2000
ROLLOUT_INTERVAL = 5000
EVAL_INTERVAL = 10_000
CLUSTER_MASK_PROB = 0.1

current_curriculum_idx = 0
state, info = env.reset()
episode_reward = 0.0
episode_steps = 0
episode_count = 0
t_train_start = time.time()

print(f"\n{'='*60}")
print(f"  Training: {TOTAL_STEPS:,} steps")
print(f"{'='*60}\n")

for step in range(1, TOTAL_STEPS + 1):

    # ── Curriculum expansion ──
    if current_curriculum_idx < len(CURRICULUM) - 1:
        if step >= CURRICULUM[current_curriculum_idx + 1]["start_step"]:
            current_curriculum_idx += 1
            new_max = CURRICULUM[current_curriculum_idx]["max_pairs"]
            env.set_max_pairs(new_max if new_max is not None else len(env._all_pairs))
            print(f"\n  *** CURRICULUM: {CURRICULUM[current_curriculum_idx]['name']} "
                  f"→ {len(env._pairs)} pairs ***\n")

    # ── Select action ──
    if step <= WARMUP_STEPS:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state, deterministic=False)

    # ── Step environment ──
    next_state, reward, terminated, truncated, step_info = env.step(action)
    done = terminated or truncated

    # ── Cluster masking (10% probability) ──
    if random.random() < CLUSTER_MASK_PROB:
        masked_next = next_state.copy()
        masked_next[2 * METRIC_DIM:] = 0.0
        masked_next[2 * METRIC_DIM + N_CLUSTERS] = 1.0  # unknown
    else:
        masked_next = next_state

    # ── Store transition ──
    replay_buffer.push(state, action, reward, masked_next, float(done))

    state = next_state
    episode_reward += reward
    episode_steps += 1

    # ── Episode boundary ──
    if done:
        state, info = env.reset()
        episode_count += 1
        episode_reward = 0.0
        episode_steps = 0

    # ── Training updates ──
    if step > WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        metrics = agent.update(batch)

    # ── Logging ──
    if step % LOG_INTERVAL == 0:
        mse = step_info.get("mse", float('nan'))
        TRAIN_LOG["rewards"].append(reward)
        TRAIN_LOG["mse_values"].append(mse)
        TRAIN_LOG["steps"].append(step)
        if step > WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
            TRAIN_LOG["q_values"].append(metrics.get("critic_loss", 0))
            TRAIN_LOG["alpha_values"].append(metrics.get("alpha", 0))

        # Action sparsity: fraction of dims with |action| < 0.01
        sparsity = float(np.mean(np.abs(action) < 0.01))
        TRAIN_LOG["action_sparsity"].append(sparsity)

        if step % (LOG_INTERVAL * 10) == 0:
            elapsed = time.time() - t_train_start
            mse_arr = TRAIN_LOG["mse_values"][-100:]
            avg_mse = np.nanmean(mse_arr) if mse_arr else float('nan')
            print(f"  step {step:>7,} | reward {reward:.4f} | mse {mse:.4f} "
                  f"(avg {avg_mse:.4f}) | alpha {metrics.get('alpha', 0):.4f} "
                  f"| sparsity {sparsity:.2%} | {elapsed:.0f}s")

    # ── Checkpoint ──
    if step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = CHECKPOINT_DIR / f"ursula_step_{step:07d}.pt"
        agent.save_checkpoint(ckpt_path, step, extra={
            "curriculum_idx": current_curriculum_idx,
            "train_log": {k: v[-100:] for k, v in TRAIN_LOG.items()},
        })
        print(f"  [CKPT] Saved: {ckpt_path.name}")

    # ── Audio rollout recording ──
    if step % ROLLOUT_INTERVAL == 0:
        try:
            obs, rinfo = env.reset()
            det_action = agent.select_action(obs, deterministic=True)
            processed = apply_plugins(env._current_audio, SR, decode_action(det_action))
            rollout_dir = OUTPUT / 'rollouts'
            rollout_dir.mkdir(exist_ok=True)
            sf.write(str(rollout_dir / f"step_{step:07d}_degraded.wav"), env._current_audio, SR)
            sf.write(str(rollout_dir / f"step_{step:07d}_processed.wav"), processed, SR)
            sf.write(str(rollout_dir / f"step_{step:07d}_reference.wav"),
                     env._reference_audio if hasattr(env, '_reference_audio') else env._current_audio, SR)
            print(f"  [AUDIO] Saved rollout at step {step}")
        except Exception as e:
            print(f"  [AUDIO] Rollout failed: {e}")

    # ── Evaluation ──
    if step % EVAL_INTERVAL == 0:
        print(f"\n  --- Evaluation at step {step} ---")
        eval_rewards = []; eval_mses = []; eval_floors = []
        for _ in range(100):
            obs, einfo = env.reset()
            for _ in range(MAX_EPISODE_STEPS):
                a = agent.select_action(obs, deterministic=True)
                obs, r, term, trunc, si = env.step(a)
                if term or trunc:
                    break
            eval_rewards.append(r)
            eval_mses.append(si.get("mse", float('nan')))
            eval_floors.append(einfo.get("identity_floor", 0.05))
        eval_rewards = np.array(eval_rewards)
        eval_mses = np.array(eval_mses)
        eval_floors = np.array(eval_floors)
        below_floor = np.mean(eval_mses < eval_floors) * 100
        below_2x = np.mean(eval_mses < 2 * eval_floors) * 100
        print(f"    Mean reward:    {eval_rewards.mean():.4f} ± {eval_rewards.std():.4f}")
        print(f"    Mean MSE:       {eval_mses.mean():.6f} ± {eval_mses.std():.6f}")
        print(f"    < floor:        {below_floor:.1f}%")
        print(f"    < 2× floor:     {below_2x:.1f}%")
        print(f"  --- End eval ---\n")

# ══════════════════════════════════════════════════════════════════════════════
# Save Final Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

final_path = CHECKPOINT_DIR / "ursula_final.pt"
agent.save_checkpoint(final_path, TOTAL_STEPS, extra={
    "curriculum_idx": current_curriculum_idx,
    "train_log": TRAIN_LOG,
})
print(f"\n  Final checkpoint saved: {final_path}")
print(f"  Total training time: {time.time() - t_train_start:.0f}s")
