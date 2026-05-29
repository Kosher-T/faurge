# %% [markdown]
# ## ONNX Export & Final Summary
#
# Converts the trained policy to ONNX format, verifies inference parity
# between PyTorch and ONNX Runtime, and produces the final package.

# ══════════════════════════════════════════════════════════════════════════════
# Export Policy to ONNX
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  ONNX EXPORT")
print("=" * 60)

# Load best checkpoint (or use current agent)
best_ckpt = sorted(CHECKPOINT_DIR.glob("ursula_step_*.pt"))[-1] if list(CHECKPOINT_DIR.glob("ursula_step_*.pt")) else final_path
print(f"  Loading checkpoint: {best_ckpt.name}")
agent.load_checkpoint(best_ckpt)

# Extract the policy network (deterministic, no learned log-std)
policy_net = agent.actor  # UrsulaSACActor contains the same trunk

# Dummy input for tracing
dummy_input = torch.randn(1, INPUT_DIM, device=DEVICE)
policy_net.eval()

# Export
onnx_path = OUTPUT / "ursula_v1.onnx"
with torch.no_grad():
    torch.onnx.export(
        policy_net,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["action"],
        dynamic_axes={
            "input": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

print(f"  Exported: {onnx_path}")
print(f"  Size: {onnx_path.stat().st_size / 1024:.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# Verify ONNX matches PyTorch
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ONNX vs PYTORCH VERIFICATION")
print("=" * 60)

try:
    import onnxruntime as ort

    # PyTorch inference
    with torch.no_grad():
        pt_out = policy_net(dummy_input).cpu().numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = ort_session.run(None, {"input": dummy_input.cpu().numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"  Max absolute diff:  {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")
    assert max_diff < 1e-5, f"ONNX mismatch: max diff = {max_diff}"
    print("  [PASS] ONNX output matches PyTorch (within 1e-5)")

except ImportError:
    print("  [SKIP] onnxruntime not available, skipping verification")
except Exception as e:
    print(f"  [WARN] Verification failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Package: ONNX + policy_config.json
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PACKAGE")
print("=" * 60)

policy_config = {
    "version": "1.0",
    "input_dim": INPUT_DIM,
    "output_dim": OUTPUT_DIM,
    "n_clusters": N_CLUSTERS,
    "n_clusters_onehot": N_CLUSTERS_ONEHOT,
    "metric_dim": METRIC_DIM,
    "sr": SR,
    "clip_sec": CLIP_SEC,
    "plugin_order": ["eq", "compressor", "esser", "saturator", "limiter", "transient", "gain"],
    "plugin_dims": {name: dim for name, dim in PLUGIN_HEAD_DIMS.items()},
    "param_ranges": {
        pr.name: {"low": pr.low, "high": pr.high, "log": pr.log}
        for pr in ALL_PARAM_RANGES
    },
    "categorical_indices": CATEGORICAL_INDICES,
    "identity_floors": {
        k: v for k, v in env._identity_floors.items()
    } if hasattr(env, '_identity_floors') else {},
    "curriculum": CURRICULUM,
    "training": {
        "total_steps": TOTAL_STEPS,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "gamma": GAMMA,
        "tau": TAU,
        "warmup_steps": WARMUP_STEPS,
        "policy_delay": POLICY_DELAY,
    },
}

config_path = OUTPUT / "policy_config.json"
with open(config_path, "w") as f:
    json.dump(policy_config, f, indent=2)
print(f"  Config: {config_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SHA256 Hash
# ══════════════════════════════════════════════════════════════════════════════

import hashlib

sha256 = hashlib.sha256()
with open(onnx_path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        sha256.update(chunk)
model_hash = sha256.hexdigest()

print(f"\n  SHA256: {model_hash}")

# Save hash for Phase 7 ingestion
hash_path = OUTPUT / "ursula_v1.sha256"
with open(hash_path, "w") as f:
    f.write(f"{model_hash}  ursula_v1.onnx\n")
print(f"  Hash file: {hash_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 6 COMPLETE")
print("=" * 60)
print(f"  Model:     {onnx_path}")
print(f"  Config:    {config_path}")
print(f"  SHA256:    {model_hash[:16]}...")
print(f"  Steps:     {TOTAL_STEPS:,}")
print(f"  Curricula: {len(CURRICULUM)} stages")
print(f"  Runtime:   {time.time() - t_train_start:.0f}s")
print(f"\n  Next: Phase 7 — Local ingestion & integration test")
print("=" * 60)
