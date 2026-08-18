# ══════════════════════════════════════════════════════════════════════════════
# Step 1bd — Evaluate (SEEN vs UNSEEN frequencies)
# ══════════════════════════════════════════════════════════════════════════════
# Two evaluations: seen frequencies (random split) vs unseen frequencies (held-out).
# This tells us if the model generalizes or just memorizes per-frequency patterns.

import matplotlib.pyplot as plt

# ── Load best model ───────────────────────────────────────────────────────────

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ── Helper function: evaluate a loader ───────────────────────────────────────

def evaluate_loader(loader, name):
    """Evaluate model on a DataLoader. Returns dict of metrics."""
    all_preds = []
    all_targets = []
    all_freqs = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    pred_freq_hz = denormalize_freq(preds[:, 0])
    true_freq_hz = denormalize_freq(targets[:, 0])
    pred_gain_db = denormalize_gain(preds[:, 1])
    true_gain_db = denormalize_gain(targets[:, 1])

    freq_abs_err = np.abs(pred_freq_hz - true_freq_hz)
    freq_pct_err = freq_abs_err / (true_freq_hz + 1e-6) * 100
    gain_abs_err = np.abs(pred_gain_db - true_gain_db)

    return {
        'name': name,
        'pred_freq_hz': pred_freq_hz,
        'true_freq_hz': true_freq_hz,
        'pred_gain_db': pred_gain_db,
        'true_gain_db': true_gain_db,
        'freq_abs_err': freq_abs_err,
        'freq_pct_err': freq_pct_err,
        'gain_abs_err': gain_abs_err,
        'n_samples': len(preds),
    }

def print_results(r):
    """Print metrics for a result dict."""
    print(f"  Freq MAE:    {np.mean(r['freq_abs_err']):>8.0f} Hz ({np.mean(r['freq_pct_err']):.1f}% of center freq)")
    print(f"  Freq median: {np.median(r['freq_abs_err']):>8.0f} Hz")
    print(f"  Freq p90:    {np.percentile(r['freq_abs_err'], 90):>8.0f} Hz")
    print(f"  Gain MAE:    {np.mean(r['gain_abs_err']):>8.2f} dB")
    print(f"  Gain ±1dB:   {np.mean(r['gain_abs_err'] <= 1.0) * 100:>7.1f}%")
    print(f"  Gain ±2dB:   {np.mean(r['gain_abs_err'] <= 2.0) * 100:>7.1f}%")

def print_freq_table(r, n_rows=20):
    """Print per-frequency breakdown."""
    unique_freqs = np.unique(r['true_freq_hz'])
    sample_freqs = unique_freqs[::max(1, len(unique_freqs) // n_rows)]

    print(f"  {'True Freq':>10} | {'Freq MAE':>10} | {'Freq %Err':>10} | {'Gain MAE':>10} | {'N':>5}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")

    for freq in sample_freqs:
        mask = r['true_freq_hz'] == freq
        if mask.sum() == 0:
            continue
        f_mae = np.mean(r['freq_abs_err'][mask])
        f_pct = np.mean(r['freq_pct_err'][mask])
        g_mae = np.mean(r['gain_abs_err'][mask])
        print(f"  {freq:>10.0f} | {f_mae:>10.0f} | {f_pct:>9.1f}% | {g_mae:>10.2f} | {mask.sum():>5}")

# ── Evaluate both sets ───────────────────────────────────────────────────────

seen = evaluate_loader(seen_test_loader, "SEEN frequencies")
unseen = evaluate_loader(unseen_test_loader, "UNSEEN frequencies")

print("═" * 70)
print("STEP 1bd EVALUATION — Frequency + Gain Prediction (160D → 2D)")
print("═" * 70)

print(f"\n{'─'*70}")
print(f"SEEN FREQUENCIES ({seen['n_samples']:,} samples)")
print(f"{'─'*70}")
print_results(seen)

print(f"\n{'─'*70}")
print(f"UNSEEN FREQUENCIES ({unseen['n_samples']:,} samples)")
print(f"{'─'*70}")
print_results(unseen)

# ── Generalization gap ───────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"GENERALIZATION GAP (seen → unseen)")
print(f"{'─'*70}")
gap_freq = np.mean(unseen['freq_abs_err']) - np.mean(seen['freq_abs_err'])
gap_pct = np.mean(unseen['freq_pct_err']) - np.mean(seen['freq_pct_err'])
gap_gain = np.mean(unseen['gain_abs_err']) - np.mean(seen['gain_abs_err'])
print(f"  Freq MAE gap:  {gap_freq:+.0f} Hz ({gap_pct:+.1f}%)")
print(f"  Gain MAE gap:  {gap_gain:+.2f} dB")

if np.mean(unseen['freq_pct_err']) < 15:
    print(f"\n✅ UNSEEN FREQUENCIES: model generalizes (< 15% error)")
elif np.mean(unseen['freq_pct_err']) < 25:
    print(f"\n⚠️  UNSEEN FREQUENCIES: moderate generalization (15-25% error)")
else:
    print(f"\n❌ UNSEEN FREQUENCIES: poor generalization (> 25% error)")

# ── Per-frequency breakdown: SEEN ────────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"SEEN FREQUENCIES — PER-FREQUENCY BREAKDOWN")
print(f"{'─'*70}")
print_freq_table(seen)

# ── Per-frequency breakdown: UNSEEN ──────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"UNSEEN FREQUENCIES — PER-FREQUENCY BREAKDOWN")
print(f"{'─'*70}")
print_freq_table(unseen)

# ── Scatter plots: seen vs unseen ────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Seen frequency scatter
ax = axes[0, 0]
ax.scatter(seen['true_freq_hz'], seen['pred_freq_hz'], alpha=0.1, s=5, c='blue')
lims = [0, max(seen['true_freq_hz'].max(), seen['pred_freq_hz'].max()) * 1.05]
ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
ax.set_xlabel('True Frequency (Hz)')
ax.set_ylabel('Predicted Frequency (Hz)')
ax.set_title(f'Seen: Freq (MAE={np.mean(seen["freq_abs_err"]):.0f}Hz)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Unseen frequency scatter
ax = axes[0, 1]
ax.scatter(unseen['true_freq_hz'], unseen['pred_freq_hz'], alpha=0.1, s=5, c='orange')
lims = [0, max(unseen['true_freq_hz'].max(), unseen['pred_freq_hz'].max()) * 1.05]
ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
ax.set_xlabel('True Frequency (Hz)')
ax.set_ylabel('Predicted Frequency (Hz)')
ax.set_title(f'Unseen: Freq (MAE={np.mean(unseen["freq_abs_err"]):.0f}Hz)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Seen gain scatter
ax = axes[1, 0]
ax.scatter(seen['true_gain_db'], seen['pred_gain_db'], alpha=0.1, s=5, c='blue')
lims = [GAIN_MIN - 1, GAIN_MAX + 1]
ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
ax.set_xlabel('True Gain (dB)')
ax.set_ylabel('Predicted Gain (dB)')
ax.set_title(f'Seen: Gain (MAE={np.mean(seen["gain_abs_err"]):.2f}dB)')
ax.legend()
ax.grid(True, alpha=0.3)

# Unseen gain scatter
ax = axes[1, 1]
ax.scatter(unseen['true_gain_db'], unseen['pred_gain_db'], alpha=0.1, s=5, c='orange')
lims = [GAIN_MIN - 1, GAIN_MAX + 1]
ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
ax.set_xlabel('True Gain (dB)')
ax.set_ylabel('Predicted Gain (dB)')
ax.set_title(f'Unseen: Gain (MAE={np.mean(unseen["gain_abs_err"]):.2f}dB)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CHECKPOINT_DIR / 'scatter_seen_unseen.png', dpi=100)
plt.show()
print(f"Scatter plot saved to: {CHECKPOINT_DIR / 'scatter_seen_unseen.png'}")

# ── Error distribution comparison ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(seen['freq_abs_err'], bins=50, alpha=0.5, label=f'Seen (μ={np.mean(seen["freq_abs_err"]):.0f}Hz)', color='blue')
axes[0].hist(unseen['freq_abs_err'], bins=50, alpha=0.5, label=f'Unseen (μ={np.mean(unseen["freq_abs_err"]):.0f}Hz)', color='orange')
axes[0].set_xlabel('Frequency Absolute Error (Hz)')
axes[0].set_ylabel('Count')
axes[0].set_title('Frequency Error: Seen vs Unseen')
axes[0].legend()

axes[1].hist(seen['gain_abs_err'], bins=50, alpha=0.5, label=f'Seen (μ={np.mean(seen["gain_abs_err"]):.2f}dB)', color='blue')
axes[1].hist(unseen['gain_abs_err'], bins=50, alpha=0.5, label=f'Unseen (μ={np.mean(unseen["gain_abs_err"]):.2f}dB)', color='orange')
axes[1].set_xlabel('Gain Absolute Error (dB)')
axes[1].set_ylabel('Count')
axes[1].set_title('Gain Error: Seen vs Unseen')
axes[1].legend()

plt.tight_layout()
plt.savefig(CHECKPOINT_DIR / 'error_dist_seen_unseen.png', dpi=100)
plt.show()
print(f"Error distribution saved to: {CHECKPOINT_DIR / 'error_dist_seen_unseen.png'}")

# ── Verdict ───────────────────────────────────────────────────────────────────

print()
print("═" * 70)
unseen_freq_pct = np.mean(unseen['freq_pct_err'])
unseen_gain_mae = np.mean(unseen['gain_abs_err'])
seen_freq_pct = np.mean(seen['freq_pct_err'])
seen_gain_mae = np.mean(seen['gain_abs_err'])

print(f"Seen:     freq={seen_freq_pct:.1f}%, gain={seen_gain_mae:.2f}dB")
print(f"Unseen:   freq={unseen_freq_pct:.1f}%, gain={unseen_gain_mae:.2f}dB")
print(f"Gap:      freq={gap_pct:+.1f}%, gain={gap_gain:+.2f}dB")

if unseen_freq_pct < 15 and unseen_gain_mae < 1.0:
    print("\n✅ PASSED — model generalizes to unseen frequencies")
elif unseen_freq_pct < 25 and unseen_gain_mae < 1.5:
    print("\n⚠️  PARTIAL — moderate generalization, needs improvement")
else:
    print("\n❌ FAILED — model does not generalize to unseen frequencies")
print("═" * 70)
