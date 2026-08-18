# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Evaluate
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

# ── Load best model ───────────────────────────────────────────────────────────

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ── Run evaluation ────────────────────────────────────────────────────────────

all_preds = []
all_targets = []
all_inputs = []

with torch.no_grad():
    for batch_degraded, batch_clean in test_loader:
        batch_degraded = batch_degraded.to(device)
        pred = model(batch_degraded)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_clean.numpy())
        all_inputs.append(batch_degraded.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
all_inputs = np.concatenate(all_inputs, axis=0)

# ── Compute metrics ───────────────────────────────────────────────────────────

# MSE: model prediction vs target
mse_pred = np.mean((all_preds - all_targets) ** 2)
# MSE: just copying degraded (baseline)
mse_baseline = np.mean((all_inputs - all_targets) ** 2)
# Improvement
improvement = (mse_baseline - mse_pred) / mse_baseline * 100

print("═" * 60)
print("EVALUATION RESULTS")
print("═" * 60)
print(f"Baseline MSE (copy degraded):  {mse_baseline:.6f}")
print(f"Model MSE (predicted clean):   {mse_pred:.6f}")
print(f"Improvement:                   {improvement:.1f}%")
print()

# Per-dimension breakdown
dim_names = [f'LTAS_{i}' for i in range(64)] + ['LUFS', 'Crest', 'ZCR']
dim_mse_baseline = np.mean((all_inputs - all_targets) ** 2, axis=0)
dim_mse_pred = np.mean((all_preds - all_targets) ** 2, axis=0)

# Show top 5 most improved and worst improved dimensions
dim_improvement = (dim_mse_baseline - dim_mse_pred) / (dim_mse_baseline + 1e-10) * 100
top_improved = np.argsort(dim_improvement)[-5:][::-1]
worst_improved = np.argsort(dim_improvement)[:5]

print("Top 5 most improved dimensions:")
for idx in top_improved:
    print(f"  {dim_names[idx]:12s} — baseline: {dim_mse_baseline[idx]:.4f} → model: {dim_mse_pred[idx]:.4f} ({dim_improvement[idx]:+.1f}%)")

print("\n5 worst/improved dimensions:")
for idx in worst_improved:
    print(f"  {dim_names[idx]:12s} — baseline: {dim_mse_baseline[idx]:.4f} → model: {dim_mse_pred[idx]:.4f} ({dim_improvement[idx]:+.1f}%)")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Loss curves
axes[0].plot(train_losses, label='Train', alpha=0.8)
axes[0].plot(test_losses, label='Test', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Target (scatter)
sample_idx = np.random.choice(len(all_preds), min(500, len(all_preds)), replace=False)
axes[1].scatter(all_targets[sample_idx, 64], all_preds[sample_idx, 64], alpha=0.3, s=10)
axes[1].plot([-30, -5], [-30, -5], 'r--', label='Perfect')
axes[1].set_xlabel('Target LUFS')
axes[1].set_ylabel('Predicted LUFS')
axes[1].set_title('LUFS: Predicted vs Target')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Per-dimension improvement
axes[2].barh(range(67), dim_improvement)
axes[2].set_yticks(range(67))
axes[2].set_yticklabels(dim_names, fontsize=6)
axes[2].set_xlabel('Improvement (%)')
axes[2].set_title('Per-Dimension Improvement')
axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(STEP0_DIR / 'evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to: {STEP0_DIR / 'evaluation.png'}")

# ── Verdict ───────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
if improvement > 10:
    print("✅ Step 0 PASSED — model learns metric space")
    print("   Ready for Step 1 (single-parameter causality)")
elif improvement > 0:
    print("⚠️  Step 0 MARGINAL — some learning, but weak")
    print("   Consider more degradations or more epochs")
else:
    print("❌ Step 0 FAILED — model doesn't learn")
    print("   Check degradation range, metric extraction, or model architecture")
print("═" * 60)
