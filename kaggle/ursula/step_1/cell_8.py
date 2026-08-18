# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Evaluate
# ══════════════════════════════════════════════════════════════════════════════
# Evaluate model's ability to predict plugin parameters.
# Reports MAE and accuracy per plugin.

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
print(f"STEP 1 EVALUATION — Substep {SUBSTEP}")
print("═" * 60)
print(f"Overall MSE:  {mse:.6f}")
print(f"Overall MAE:  {mae:.6f}")
print()

# ── Per-plugin breakdown ─────────────────────────────────────────────────────

dim_idx = 0

if USE_GAIN:
    gain_pred = all_preds[:, dim_idx]
    gain_true = all_targets[:, dim_idx]
    gain_mae = np.mean(np.abs(gain_pred - gain_true))
    gain_within_1 = np.mean(np.abs(gain_pred - gain_true) <= 1.0) * 100
    gain_within_2 = np.mean(np.abs(gain_pred - gain_true) <= 2.0) * 100

    print("GAIN")
    print(f"  MAE:          {gain_mae:.2f} dB")
    print(f"  Within ±1dB:  {gain_within_1:.1f}%")
    print(f"  Within ±2dB:  {gain_within_2:.1f}%")
    print(f"  True range:   [{gain_true.min():.1f}, {gain_true.max():.1f}] dB")
    print(f"  Pred range:   [{gain_pred.min():.1f}, {gain_pred.max():.1f}] dB")
    print()
    dim_idx += 1

if USE_EQ:
    for b in range(N_EQ_BANDS):
        freq_log_pred = all_preds[:, dim_idx]
        freq_log_true = all_targets[:, dim_idx]
        gain_pred = all_preds[:, dim_idx + 1]
        gain_true = all_targets[:, dim_idx + 1]
        q_pred = all_preds[:, dim_idx + 2]
        q_true = all_targets[:, dim_idx + 2]

        # Convert log(freq) back to linear Hz for evaluation
        freq_pred = np.exp(freq_log_pred)
        freq_true = np.exp(freq_log_true)

        freq_mae = np.mean(np.abs(freq_pred - freq_true))
        gain_mae = np.mean(np.abs(gain_pred - gain_true))
        q_mae = np.mean(np.abs(q_pred - q_true))
        freq_within_50 = np.mean(np.abs(freq_pred - freq_true) <= 50.0) * 100
        gain_within_1 = np.mean(np.abs(gain_pred - gain_true) <= 1.0) * 100

        print(f"EQ BAND {b+1}")
        print(f"  Freq MAE:     {freq_mae:.0f} Hz")
        print(f"  Freq ±50Hz:   {freq_within_50:.1f}%")
        print(f"  Gain MAE:     {gain_mae:.2f} dB")
        print(f"  Gain ±1dB:    {gain_within_1:.1f}%")
        print(f"  Q MAE:        {q_mae:.2f}")
        print()
        dim_idx += 3

if USE_COMPRESSOR:
    comp_names = ["threshold_db", "ratio", "attack_ms", "release_ms"]
    comp_units = ["dB", "", "ms", "ms"]
    for i, (name, unit) in enumerate(zip(comp_names, comp_units)):
        pred = all_preds[:, dim_idx + i]
        true = all_targets[:, dim_idx + i]
        mae = np.mean(np.abs(pred - true))
        print(f"  {name}: MAE = {mae:.2f} {unit}")
    dim_idx += 4

# ── Verdict ───────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
if USE_GAIN and gain_mae < 1.0:
    print("✅ Step 1 PASSED — model learns gain causality")
    print("   Ready for next substep")
elif not USE_GAIN:
    print("⚠️  Check per-plugin metrics above")
else:
    print("⚠️  Step 1 MARGINAL — gain MAE > 1dB")
    print("   Consider more degradations or epochs")
print("═" * 60)
