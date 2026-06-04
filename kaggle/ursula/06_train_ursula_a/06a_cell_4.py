# %% [markdown]
# ## Save Trained Weights & Verify on Test Set
#
# Save the best policy weights found during training.
# Verify that the trained policy produces reasonable outputs
# by applying it to TEST SET pairs (unseen during training) and
# measuring the resulting audio-domain MSE.

# ══════════════════════════════════════════════════════════════════════════════
# Save trained policy weights
# ══════════════════════════════════════════════════════════════════════════════

policy.eval()
pretrained_path = OUTPUT / "ursula_sl_v1.pt"
torch.save({
    "policy_state_dict": policy.state_dict(),
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "training": {
        "supervised_epochs": SUPERVISED_EPOCHS,
        "n_pairs_total": len(supervised_data),
        "n_pairs_train": len(train_indices),
        "n_pairs_test": len(test_indices),
        "method": "inverse_degradation",
        "best_test_mse": float(best_test_mse),
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
    },
    "train_indices": train_indices.tolist(),
    "test_indices": test_indices.tolist(),
}, pretrained_path)

print(f"  Saved trained weights: {pretrained_path}")
print(f"  Size: {pretrained_path.stat().st_size / 1024:.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# Verify: run trained policy on TEST SET pairs (audio-domain evaluation)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  VERIFICATION: Policy on TEST SET pairs (audio-domain)")
print(f"{'='*60}")

import soundfile as sf

# Use test set indices for verification — these were NEVER seen during training
verify_test_indices = test_indices[:min(50, len(test_indices))]
policy_mses = []
initial_mses = []

with torch.no_grad():
    for vi, ti in enumerate(verify_test_indices):
        pd = pair_data[ti]
        oh = np.zeros(N_CLUSTERS_ONEHOT, dtype=np.float32)
        if 0 <= pd['cluster_id'] < N_CLUSTERS:
            oh[pd['cluster_id']] = 1.0
        else:
            oh[N_CLUSTERS] = 1.0
        obs = np.concatenate([pd['m_degraded'], pd['m_reference'], oh]).astype(np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Predict action
        action = policy(obs_t).squeeze(0).cpu().numpy()

        # Apply to degraded audio and measure audio-domain MSE
        try:
            plugin_dicts = decode_action(action)
            processed = apply_plugins(pd['degraded_audio'], SR, plugin_dicts)
            m_result = extract_metrics_67d(processed)
            result_mse = float(np.mean((m_result - pd['m_reference']) ** 2))
            initial_mse = float(np.mean((pd['m_degraded'] - pd['m_reference']) ** 2))
        except Exception as e:
            result_mse = float('nan')
            initial_mse = float('nan')
            print(f"  Test pair {vi}: ERROR — {e}")
            continue

        policy_mses.append(result_mse)
        initial_mses.append(initial_mse)

        improved = "✅" if result_mse < initial_mse else "❌"
        if vi < 10 or (vi + 1) % 25 == 0:
            print(f"  Test pair {vi:>3} ({pd['pair_id']}): "
                  f"initial={initial_mse:.2f} → policy={result_mse:.2f} "
                  f"{improved} | action_norm={np.linalg.norm(action):.3f}")

        # Save audio for first few test pairs
        if vi < 5:
            roll_dir = OUTPUT / 'test_rollouts'
            roll_dir.mkdir(exist_ok=True)
            sf.write(str(roll_dir / f"test_{vi}_degraded.wav"), pd['degraded_audio'], SR)
            sf.write(str(roll_dir / f"test_{vi}_processed.wav"), processed, SR)
            sf.write(str(roll_dir / f"test_{vi}_reference.wav"), pd['reference_audio'], SR)

# Summary
policy_mses = np.array(policy_mses)
initial_mses = np.array(initial_mses)
n_improved = int(np.sum(policy_mses < initial_mses))
pct_improved = n_improved / len(policy_mses) * 100 if len(policy_mses) > 0 else 0
avg_reduction = float(np.mean(initial_mses - policy_mses)) if len(policy_mses) > 0 else 0

print(f"\n{'='*60}")
print(f"  VERIFICATION SUMMARY (TEST SET — unseen during training)")
print(f"{'='*60}")
print(f"  Evaluated: {len(policy_mses)} test pairs")
print(f"  Initial MSE (degraded vs ref):  {initial_mses.mean():.2f} ± {initial_mses.std():.2f}")
print(f"  Policy MSE (processed vs ref):  {policy_mses.mean():.2f} ± {policy_mses.std():.2f}")
print(f"  Pairs improved: {n_improved}/{len(policy_mses)} ({pct_improved:.1f}%)")
print(f"  Avg MSE reduction: {avg_reduction:.2f}")
if initial_mses.mean() > 0:
    print(f"  Relative improvement: {avg_reduction / initial_mses.mean() * 100:.1f}%")
print(f"{'='*60}")
print(f"  PHASE 6A COMPLETE")
print(f"{'='*60}")
print(f"  Trained weights: {pretrained_path}")
print(f"  Loss curves:     {loss_history_path}")
print(f"{'='*60}")
