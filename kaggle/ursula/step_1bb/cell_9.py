# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Evaluate on held-out frequencies
# ══════════════════════════════════════════════════════════════════════════════
# Test on unseen frequencies (not in curriculum) to check generalization.

import matplotlib.pyplot as plt

# ── Load best model ───────────────────────────────────────────────────────────

model.load_state_dict(torch.load(CHECKPOINT_DIR / 'curriculum_best.pt'))
model.eval()

# ── Held-out test frequencies ────────────────────────────────────────────────

UNSEEN_FREQS = [175, 350, 700, 1400, 2800, 5500, 8000, 12000]

print("═" * 60)
print("CROSS-FREQUENCY GENERALIZATION (unseen frequencies)")
print("═" * 60)

unseen_results = []

for freq_hz in UNSEEN_FREQS:
    freq_val = freq_to_input(freq_hz)
    test_inputs = []
    test_labels = []

    for clip in clean_clips:
        for _ in range(200):
            degraded, gain_db = degrade_with_eq(clip, freq_hz)
            degraded = np.clip(degraded, -1.0, 1.0)
            deg_m = extract_metrics_67d(degraded)
            clean_m = extract_metrics_67d(clip)
            inp = np.concatenate([deg_m, clean_m, [freq_val]])
            test_inputs.append(inp)
            test_labels.append([gain_db])

    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)

    test_t = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(test_t).cpu().numpy()

    mae = np.mean(np.abs(preds - test_labels))
    within_1 = np.mean(np.abs(preds - test_labels) <= 1.0) * 100

    unseen_results.append({'freq_hz': freq_hz, 'mae': float(mae), 'within_1dB': float(within_1)})
    print(f"  {freq_hz:>6}Hz (UNSEEN) → MAE={mae:.2f}dB, ±1dB={within_1:.0f}%")

# ── Compare with seen frequencies ────────────────────────────────────────────

seen_mae = np.mean([r['mae'] for r in results])
unseen_mae = np.mean([r['mae'] for r in unseen_results])
print(f"\nSeen avg MAE:     {seen_mae:.2f} dB")
print(f"Unseen avg MAE:   {unseen_mae:.2f} dB")
print(f"Unseen/Seen ratio: {unseen_mae / seen_mae:.2f}x")

print()
if unseen_mae < 0.5:
    print("✅ Model generalizes — frequency input works across spectrum")
elif unseen_mae < 1.0:
    print("⚠️  Partial generalization")
else:
    print("❌ No generalization — model overfits to curriculum frequencies")

with open(CHECKPOINT_DIR / 'unseen_freq.json', 'w') as f:
    json.dump({'seen_results': results, 'unseen_results': unseen_results}, f, indent=2)
print(f"\nSaved to: {CHECKPOINT_DIR / 'unseen_freq.json'}")
