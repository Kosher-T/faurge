# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Evaluate (Gain-Only EQ)
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

# ── Load best model ───────────────────────────────────────────────────────────

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ── Run evaluation ────────────────────────────────────────────────────────────

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(device)
        pred = model(batch_inputs)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_labels.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# ── Overall metrics ───────────────────────────────────────────────────────────

mse = np.mean((all_preds - all_targets) ** 2)
mae = np.mean(np.abs(all_preds - all_targets))

print("═" * 60)
print("STEP 1ba EVALUATION — EQ Causality (Gain-Only)")
print(f"Fixed EQ: {EQ_FIXED_FREQ_HZ}Hz, Q={EQ_FIXED_Q}")
print("═" * 60)
print(f"Overall MSE:  {mse:.6f}")
print(f"Overall MAE:  {mae:.2f} dB")
print()

# ── Gain accuracy ─────────────────────────────────────────────────────────────

gain_within_1 = np.mean(np.abs(all_preds - all_targets) <= 1.0) * 100
gain_within_2 = np.mean(np.abs(all_preds - all_targets) <= 2.0) * 100
gain_within_3 = np.mean(np.abs(all_preds - all_targets) <= 3.0) * 100

print(f"Gain ±1dB:  {gain_within_1:.1f}%")
print(f"Gain ±2dB:  {gain_within_2:.1f}%")
print(f"Gain ±3dB:  {gain_within_3:.1f}%")
print()

# ── Scatter plot ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(all_targets, all_preds, alpha=0.3, s=10)
lims = [min(all_targets.min(), all_preds.min()) - 1, max(all_targets.max(), all_preds.max()) + 1]
ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
ax.set_xlabel('True Gain (dB)')
ax.set_ylabel('Predicted Gain (dB)')
ax.set_title(f'EQ Gain Prediction (MAE={mae:.2f}dB)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHECKPOINT_DIR / 'scatter.png', dpi=100)
plt.show()
print(f"Plot saved to: {CHECKPOINT_DIR / 'scatter.png'}")

# ── Verdict ───────────────────────────────────────────────────────────────────

print()
print("═" * 60)
if mae < 1.0:
    print("✅ Step 1ba PASSED — model learns EQ gain causality")
elif mae < 2.0:
    print("⚠️  Step 1ba MARGINAL — some learning, but weak")
else:
    print("❌ Step 1ba FAILED — model doesn't learn EQ gain")
print("═" * 60)
