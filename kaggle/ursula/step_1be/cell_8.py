# ══════════════════════════════════════════════════════════════════════════════
# Step 1be — Phase 2: Train + Evaluate (2×2 Matrix)
# ══════════════════════════════════════════════════════════════════════════════
# Train EQ prediction model on multiple clips, multiple speakers.
# Evaluate on 2×2 matrix: seen/unseen clips × seen/unseen freqs.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

# ── Load normalizer ──────────────────────────────────────────────────────────
with open(NORMALIZER_PATH) as f:
    normalizer = json.load(f)
metric_min = np.array(normalizer['min'], dtype=np.float32)
metric_range = np.array(normalizer['range'], dtype=np.float32)

def denormalize_metrics(x):
    return x * metric_range + metric_min

# ── DataLoaders ──────────────────────────────────────────────────────────────
class EQDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

n = len(p2_inputs)
idx = np.random.permutation(n)
split = int(n * TRAIN_SPLIT)

train_dataset = EQDataset(p2_inputs[idx[:split]], p2_targets[idx[:split]])
test_dataset = EQDataset(p2_inputs[idx[split:]], p2_targets[idx[split:]])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Phase 2 — EQ Prediction")
print(f"Train: {len(train_dataset):,} | Test: {len(test_dataset):,}")

# ── Train ────────────────────────────────────────────────────────────────────
optimizer = optim.Adam(eq_model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_test_loss = float('inf')
patience_ct = 0

print(f"\nTraining: {EPOCHS} epochs, patience={PATIENCE}")
for epoch in range(EPOCHS):
    eq_model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        pred = eq_model(bx)

        # Weighted loss: 0.35 freq + 0.35 eq_gain + 0.30 global_gain
        loss = (FREQ_LOSS_WEIGHT * criterion(pred[:, 0], by[:, 0]) +
                EQ_GAIN_LOSS_WEIGHT * criterion(pred[:, 1], by[:, 1]) +
                GLOBAL_GAIN_LOSS_WEIGHT * criterion(pred[:, 2], by[:, 2]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(bx)
    train_loss /= len(train_dataset)

    eq_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = eq_model(bx)
            loss = (FREQ_LOSS_WEIGHT * criterion(pred[:, 0], by[:, 0]) +
                    EQ_GAIN_LOSS_WEIGHT * criterion(pred[:, 1], by[:, 1]) +
                    GLOBAL_GAIN_LOSS_WEIGHT * criterion(pred[:, 2], by[:, 2]))
            test_loss += loss.item() * len(bx)
    test_loss /= len(test_dataset)
    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_ct = 0
        torch.save(eq_model.state_dict(), PHASE2_MODEL_PATH)
    else:
        patience_ct += 1

    lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d} | train: {train_loss:.6f} | test: {test_loss:.6f} | best: {best_test_loss:.6f} | lr: {lr:.1e}")

    if patience_ct >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch+1}")
        break

print(f"\nBest test loss: {best_test_loss:.6f}")

# ── Evaluate on 2×2 matrix ──────────────────────────────────────────────────
eq_model.load_state_dict(torch.load(PHASE2_MODEL_PATH))
eq_model.eval()

def evaluate_eq(preds, labels, label_str):
    """Evaluate EQ predictions: freq MAE, ±15Hz, gain MAE, ±1dB, global MAE."""
    freq_pred_hz = denormalize_freq(preds[:, 0])
    freq_true_hz = denormalize_freq(labels[:, 0])
    gain_pred_db = denormalize_eq_gain(preds[:, 1])
    gain_true_db = denormalize_eq_gain(labels[:, 1])
    global_pred_db = denormalize_global_gain(preds[:, 2])
    global_true_db = denormalize_global_gain(labels[:, 2])

    freq_mae = np.mean(np.abs(freq_pred_hz - freq_true_hz))
    freq_pct = freq_mae / np.mean(freq_true_hz) * 100
    within_15hz = np.mean(np.abs(freq_pred_hz - freq_true_hz) <= 15.0) * 100

    gain_mae = np.mean(np.abs(gain_pred_db - gain_true_db))
    within_1db = np.mean(np.abs(gain_pred_db - gain_true_db) <= 1.0) * 100

    global_mae = np.mean(np.abs(global_pred_db - global_true_db))
    within_1db_global = np.mean(np.abs(global_pred_db - global_true_db) <= 1.0) * 100

    print(f"\n  {label_str}")
    print(f"    Freq:   MAE={freq_mae:.1f} Hz ({freq_pct:.1f}%) | ±15Hz: {within_15hz:.1f}%")
    print(f"    Gain:   MAE={gain_mae:.2f} dB | ±1dB: {within_1db:.1f}%")
    print(f"    Global: MAE={global_mae:.2f} dB | ±1dB: {within_1db_global:.1f}%")

    return {
        'freq_mae': freq_mae, 'freq_pct': freq_pct, 'within_15hz': within_15hz,
        'gain_mae': gain_mae, 'within_1db': within_1db,
        'global_mae': global_mae, 'within_1db_global': within_1db_global,
    }

# Run eval on all 4 datasets
eval_datasets = [
    ('eval_seen_seen', eval_seen_seen, 'Seen clip × Seen freq'),
    ('eval_seen_unseen', eval_seen_unseen, 'Seen clip × Unseen freq'),
    ('eval_unseen_seen', eval_unseen_seen, 'Unseen clip × Seen freq'),
    ('eval_unseen_unseen', eval_unseen_unseen, 'Unseen clip × Unseen freq'),
]

results = {}
for name, (inputs, labels), label_str in eval_datasets:
    with torch.no_grad():
        inp_t = torch.tensor(inputs, dtype=torch.float32).to(device)
        preds = eq_model(inp_t).cpu().numpy()
    results[name] = evaluate_eq(preds, labels, label_str)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PHASE 2 EVALUATION — Multi-Clip EQ Prediction (2×2 Matrix)")
print(f"{'='*60}")

all_ok = True
for name, r in results.items():
    status = "OK" if r['within_15hz'] > 50 and r['within_1db'] > 50 else "WEAK"
    if status == "WEAK":
        all_ok = False

if all_ok:
    print("✅ PASSED — model predicts EQ parameters across clips and speakers")
else:
    print("⚠️  Some eval conditions are weak — review per-condition results above")
print(f"{'='*60}")
