# %% [markdown]
# ## Save Pretrained Weights & Verify
#
# Save the policy weights for warm-starting RL training.
# Verify that the pretrained policy produces reasonable outputs
# by applying it to each pair and measuring the resulting MSE.

# ══════════════════════════════════════════════════════════════════════════════
# Save pretrained policy weights
# ══════════════════════════════════════════════════════════════════════════════

policy.eval()
pretrained_path = OUTPUT / "ursula_pretrained.pt"
torch.save({
    "policy_state_dict": policy.state_dict(),
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "training": {
        "supervised_epochs": SUPERVISED_EPOCHS,
        "n_pairs": len(supervised_data),
        "method": "inverse_degradation",
        "avg_target_mse": float(np.mean([d[2] for d in supervised_data])),
        "best_eval_mse": float(best_eval_mse),
    },
}, pretrained_path)

print(f"  Saved pretrained weights: {pretrained_path}")
print(f"  Size: {pretrained_path.stat().st_size / 1024:.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# Verify: run pretrained policy on each pair
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  VERIFICATION: Pretrained policy on each pair")
print(f"{'='*60}")

import soundfile as sf

# Use a subset for verification (all pairs can be slow)
verify_pairs = pair_data[:min(50, len(pair_data))]
policy_mses = []
target_mses = []

with torch.no_grad():
    for pi, pd in enumerate(verify_pairs):
        oh = np.zeros(N_CLUSTERS_ONEHOT, dtype=np.float32)
        if 0 <= pd['cluster_id'] < N_CLUSTERS:
            oh[pd['cluster_id']] = 1.0
        else:
            oh[N_CLUSTERS] = 1.0
        obs = np.concatenate([pd['m_degraded'], pd['m_reference'], oh]).astype(np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Predict action
        action = policy(obs_t).squeeze(0).cpu().numpy()

        # Apply to degraded audio
        try:
            plugin_dicts = decode_action(action)
            processed = apply_plugins(pd['degraded_audio'], SR, plugin_dicts)
            m_result = extract_metrics_67d(processed)
            result_mse = float(np.mean((m_result - pd['m_reference']) ** 2))
        except Exception as e:
            result_mse = float('nan')
            print(f"  Pair {pi}: ERROR — {e}")
            continue

        target_mse = supervised_data[pi][2]
        policy_mses.append(result_mse)
        target_mses.append(target_mse)

        if pi < 10 or (pi + 1) % 25 == 0:
            print(f"  Pair {pi:>3} ({pd['pair_id']}): "
                  f"target_mse={target_mse:.2f} | policy_mse={result_mse:.2f} "
                  f"| action_norm={np.linalg.norm(action):.3f}")

        # Save audio for first few pairs
        if pi < 5:
            roll_dir = OUTPUT / 'pretrained_rollouts'
            roll_dir.mkdir(exist_ok=True)
            sf.write(str(roll_dir / f"pair_{pi}_degraded.wav"), pd['degraded_audio'], SR)
            sf.write(str(roll_dir / f"pair_{pi}_processed.wav"), processed, SR)
            sf.write(str(roll_dir / f"pair_{pi}_reference.wav"), pd['reference_audio'], SR)

# Summary
policy_mses = np.array(policy_mses)
target_mses = np.array(target_mses)
print(f"\n{'='*60}")
print(f"  VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"  Evaluated: {len(policy_mses)} pairs")
print(f"  Target MSE (inverse degradation): {target_mses.mean():.2f} ± {target_mses.std():.2f}")
print(f"  Policy MSE (neural net output):    {policy_mses.mean():.2f} ± {policy_mses.std():.2f}")
print(f"  MSE ratio (policy/target):         {(policy_mses.mean() / max(target_mses.mean(), 1e-6)):.3f}")
print(f"{'='*60}")
print(f"  PHASE 6A COMPLETE")
print(f"{'='*60}")
print(f"  Pretrained weights: {pretrained_path}")
print(f"  Next: Phase 6 (RL) — load ursula_pretrained.pt as warm start")
print(f"{'='*60}")
