# %% [markdown]
# ## Save Pretrained Weights & Verify
#
# Save the policy weights for warm-starting RL training.
# Verify that the pretrained policy produces reasonable outputs.

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
        "n_candidates": N_RANDOM_CANDIDATES,
        "avg_target_mse": float(np.mean([d[2] for d in supervised_data])),
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

with torch.no_grad():
    for pi, pd in enumerate(pair_data):
        # Build observation
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
        print(f"  Pair {pi} ({pd['pair_id']}):")
        print(f"    Target MSE (from search): {target_mse:.2f}")
        print(f"    Policy MSE (inference):   {result_mse:.2f}")
        print(f"    Action norm: {np.linalg.norm(action):.3f}")

        # Save audio
        roll_dir = OUTPUT / 'pretrained_rollouts'
        roll_dir.mkdir(exist_ok=True)
        sf.write(str(roll_dir / f"pair_{pi}_degraded.wav"), pd['degraded_audio'], SR)
        sf.write(str(roll_dir / f"pair_{pi}_processed.wav"), processed, SR)
        sf.write(str(roll_dir / f"pair_{pi}_reference.wav"), pd['reference_audio'], SR)

print(f"\n{'='*60}")
print(f"  PHASE 6A COMPLETE")
print(f"{'='*60}")
print(f"  Pretrained weights: {pretrained_path}")
print(f"  Next: Phase 6 (RL) — load ursula_pretrained.pt as warm start")
print(f"{'='*60}")
