# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Cross-Frequency Generalization Test
# ══════════════════════════════════════════════════════════════════════════════
# Train on ONE frequency, test on UNSEEN frequencies.
# Does the model learn "EQ gain" (general) or "1kHz gain" (specific)?

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

TRAIN_FREQ = 1000       # Train only on this
TEST_FREQS = [200, 500, 750, 1500, 3000, 5000, 7500, 10000, 15000]  # Test on these (including unseen)

N_DEGRADATIONS = 500
TRAIN_EPOCHS = 150
TRAIN_PATIENCE = 50

# ── Generate training data (1kHz only) ───────────────────────────────────────

print(f"Training on {TRAIN_FREQ}Hz only...")

train_clean = []
train_degraded = []
train_labels = []

for clip in clean_clips:
    clean_m = extract_metrics_67d(clip)
    for _ in range(N_DEGRADATIONS):
        gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
        degraded, _ = eq_plugin.process(
            clip, sr=SR,
            bands=[{"freq_hz": float(TRAIN_FREQ), "gain_db": float(gain_db),
                    "q": EQ_FIXED_Q, "filter_type": "peak"}]
        )
        degraded = np.clip(degraded, -1.0, 1.0)
        deg_m = extract_metrics_67d(degraded)
        train_clean.append(clean_m)
        train_degraded.append(deg_m)
        train_labels.append([gain_db])

train_clean = np.array(train_clean)
train_degraded = np.array(train_degraded)
train_labels = np.array(train_labels)

train_inputs = np.concatenate([train_degraded, train_clean], axis=1)
n = len(train_inputs)
idx = np.random.permutation(n)
split = int(n * 0.8)

train_x = torch.tensor(train_inputs[idx[:split]], dtype=torch.float32)
train_y = torch.tensor(train_labels[idx[:split]], dtype=torch.float32)
val_x = torch.tensor(train_inputs[idx[split:]], dtype=torch.float32)
val_y = torch.tensor(train_labels[idx[split:]], dtype=torch.float32)

train_ds = torch.utils.data.TensorDataset(train_x, train_y)
val_ds = torch.utils.data.TensorDataset(val_x, val_y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

# ── Train ─────────────────────────────────────────────────────────────────────

model = CausalityPredictor(output_dim=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

best_loss = float('inf')
patience_ct = 0

print(f"Training for up to {TRAIN_EPOCHS} epochs...")
for epoch in range(TRAIN_EPOCHS):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            val_loss += criterion(pred, by).item() * len(bx)
    val_loss /= len(val_ds)
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_ct = 0
        torch.save(model.state_dict(), CHECKPOINT_DIR / 'freq_model.pt')
    else:
        patience_ct += 1

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d} | val: {val_loss:.4f} | best: {best_loss:.4f}")

    if patience_ct >= TRAIN_PATIENCE:
        print(f"  Early stop at epoch {epoch+1}")
        break

print(f"Training done. Best val loss: {best_loss:.4f}")

# ── Test on all frequencies ───────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"CROSS-FREQUENCY TEST (trained on {TRAIN_FREQ}Hz)")
print(f"{'='*60}")

model.load_state_dict(torch.load(CHECKPOINT_DIR / 'freq_model.pt'))
model.eval()

results = []

for test_freq in TEST_FREQS:
    test_clean = []
    test_degraded = []
    test_labels = []

    for clip in clean_clips:
        clean_m = extract_metrics_67d(clip)
        for _ in range(200):  # 200 test samples per frequency
            gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
            degraded, _ = eq_plugin.process(
                clip, sr=SR,
                bands=[{"freq_hz": float(test_freq), "gain_db": float(gain_db),
                        "q": EQ_FIXED_Q, "filter_type": "peak"}]
            )
            degraded = np.clip(degraded, -1.0, 1.0)
            deg_m = extract_metrics_67d(degraded)
            test_clean.append(clean_m)
            test_degraded.append(deg_m)
            test_labels.append([gain_db])

    test_inputs = np.concatenate([
        np.array(test_degraded), np.array(test_clean)
    ], axis=1)
    test_labels = np.array(test_labels)

    test_t = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(test_t).cpu().numpy()

    mae = np.mean(np.abs(preds - test_labels))
    within_1 = np.mean(np.abs(preds - test_labels) <= 1.0) * 100

    is_seen = test_freq == TRAIN_FREQ
    label = "SEEN" if is_seen else "UNSEEN"
    results.append({
        'freq_hz': test_freq,
        'mae': float(mae),
        'within_1dB': float(within_1),
        'seen': is_seen,
    })
    print(f"  {test_freq:>6}Hz ({label:>6}) → MAE={mae:.2f}dB, ±1dB={within_1:.0f}%")

# ── Summary ───────────────────────────────────────────────────────────────────

seen_mae = np.mean([r['mae'] for r in results if r['seen']])
unseen_mae = np.mean([r['mae'] for r in results if not r['seen']])
print(f"\nSeen avg MAE:     {seen_mae:.2f} dB")
print(f"Unseen avg MAE:   {unseen_mae:.2f} dB")
print(f"Unseen/Seen ratio: {unseen_mae/seen_mae:.2f}x")

print()
if unseen_mae < 0.5:
    print("✅ Model generalizes — learned frequency-independent EQ gain")
elif unseen_mae < 1.0:
    print("⚠️  Partial generalization — some transfer across frequencies")
else:
    print("❌ No generalization — model only knows training frequency")

with open(CHECKPOINT_DIR / 'cross_freq.json', 'w') as f:
    json.dump({'train_freq': TRAIN_FREQ, 'results': results}, f, indent=2)
