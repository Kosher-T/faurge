# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — DataLoaders + Model + Train + Evaluate
# ══════════════════════════════════════════════════════════════════════════════
# All-in-one: loaders, 81D→81D MLP, train, evaluate.
# Pretrained from step 1be. EQ degradation forces Tier 1 learning.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

# ── Load normalizer ───────────────────────────────────────────────────────────

with open(DATA_DIR / 'normalizer.json') as f:
    normalizer = json.load(f)
metric_min = np.array(normalizer['min'], dtype=np.float32)
metric_max = np.array(normalizer['max'], dtype=np.float32)
metric_range = np.array(normalizer['range'], dtype=np.float32)

def denormalize(x):
    """Convert [0,1] back to original scale."""
    return x * metric_range + metric_min

# ── DataLoaders ───────────────────────────────────────────────────────────────

class MetricDataset(Dataset):
    def __init__(self, degraded, clean):
        self.degraded = torch.tensor(degraded, dtype=torch.float32)
        self.clean = torch.tensor(clean, dtype=torch.float32)

    def __len__(self):
        return len(self.degraded)

    def __getitem__(self, idx):
        return self.degraded[idx], self.clean[idx]

n = len(all_degraded_norm)
idx = np.random.permutation(n)
split = int(n * TRAIN_SPLIT)

train_dataset = MetricDataset(all_degraded_norm[idx[:split]], all_clean_norm[idx[:split]])
test_dataset = MetricDataset(all_degraded_norm[idx[split:]], all_clean_norm[idx[split:]])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset):,} | Test: {len(test_dataset):,}")
print(f"Data range: [{train_dataset.degraded.min():.3f}, {train_dataset.degraded.max():.3f}]")

# ── Model ─────────────────────────────────────────────────────────────────────

class MetricPredictor(nn.Module):
    """81D degraded → 81D clean metrics."""
    def __init__(self, dim=METRIC_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

model = MetricPredictor().to(device)

# ── Load pretrained weights from step 1be ─────────────────────────────────────
if PRETRAINED_PATH.exists():
    state = torch.load(PRETRAINED_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded pretrained weights from: {PRETRAINED_PATH}")
else:
    print(f"WARNING: Pretrained path not found: {PRETRAINED_PATH}")
    print("Training from scratch.")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")
print(f"Input:  {METRIC_DIM}D | Output: {METRIC_DIM}D")

# ── Train ─────────────────────────────────────────────────────────────────────

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_test_loss = float('inf')
patience_ct = 0

print(f"\nTraining: {EPOCHS} epochs, patience={PATIENCE}")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(bx)
    train_loss /= len(train_dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            test_loss += criterion(pred, by).item() * len(bx)
    test_loss /= len(test_dataset)
    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_ct = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_ct += 1

    lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d} | train: {train_loss:.6f} | test: {test_loss:.6f} | best: {best_test_loss:.6f} | lr: {lr:.1e}")

    if patience_ct >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch+1}")
        break

print(f"\nBest test loss: {best_test_loss:.6f}")

# ── Evaluate (denormalize for real MAE) ──────────────────────────────────────

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
all_targets = []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        pred = model(bx)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(by.numpy())

preds_norm = np.concatenate(all_preds)
targets_norm = np.concatenate(all_targets)

preds = denormalize(preds_norm)
targets = denormalize(targets_norm)

mae = np.mean(np.abs(preds - targets))
per_dim_mae = np.mean(np.abs(preds - targets), axis=0)

print(f"\n{'='*60}")
print(f"STEP 2a EVALUATION — Tier 1 Metric Literacy (EQ Degradation)")
print(f"{'='*60}")
print(f"Overall MAE: {mae:.6f}")

dim_names = (
    [f'LTAS_{i}' for i in range(64)] +
    ['LUFS', 'Crest', 'ZCR'] +
    ['Centroid', 'Bandwidth', 'Flatness', 'Flux', 'Rolloff', 'Skewness', 'Kurtosis', 'Slope'] +
    ['Sub', 'LowMid', 'Mid', 'Presence', 'Air'] +
    ['RMS']
)

# ── All 13 Tier 1 features (not just top 10) ────────────────────────────────
print(f"\nTier 1 features (all 13):")
tier1_names = ['Centroid', 'Bandwidth', 'Flatness', 'Flux', 'Rolloff',
               'Skewness', 'Kurtosis', 'Slope', 'Sub', 'LowMid', 'Mid', 'Presence', 'Air']
tier1_mae_vals = per_dim_mae[67:80]
for name, mae_val in zip(tier1_names, tier1_mae_vals):
    flag = " ← worst" if mae_val > 1.0 else ""
    print(f"  {name:>12}: {mae_val:.4f}{flag}")

print(f"\nPer-dimension MAE (top 10 worst overall):")
worst_idx = np.argsort(per_dim_mae)[::-1][:10]
for i in worst_idx:
    print(f"  {dim_names[i]:>12}: {per_dim_mae[i]:.4f}")

# ── Tier breakdown ──────────────────────────────────────────────────────────
tier0_mae = np.mean(per_dim_mae[:67])
tier1_mae = np.mean(per_dim_mae[67:80])
rms_mae = per_dim_mae[80]
print(f"\nTier 0 (67D) MAE: {tier0_mae:.4f}")
print(f"Tier 1 (13D) MAE: {tier1_mae:.4f}")
print(f"RMS (1D) MAE:     {rms_mae:.4f}")

# ── Pass criteria ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
tier0_ok = tier0_mae < 0.05
tier1_ok = tier1_mae < 1.0
rms_ok = rms_mae < 0.1
worst_4 = ['Rolloff', 'Bandwidth', 'Centroid', 'Kurtosis']
worst_4_mae = [per_dim_mae[dim_names.index(n)] for n in worst_4]
worst_4_ok = all(m < 1.0 for m in worst_4_mae)

if tier0_ok and tier1_ok and rms_ok and worst_4_ok:
    print("✅ PASSED — Tier 1 learned, no Tier 0 regression")
elif tier1_ok:
    print("⚠️  MARGINAL — Tier 1 improved but some issues remain")
else:
    print("❌ FAILED — Tier 1 features not sufficiently learned")
print(f"  Tier 0: {'OK' if tier0_ok else 'REGRESSED'} ({tier0_mae:.4f})")
print(f"  Tier 1: {'OK' if tier1_ok else 'WEAK'} ({tier1_mae:.4f})")
print(f"  RMS:    {'OK' if rms_ok else 'WEAK'} ({rms_mae:.4f})")
print(f"  Worst 4 (Rolloff/Bandwidth/Centroid/Kurtosis): {'OK' if worst_4_ok else 'WEAK'}")
for n, m in zip(worst_4, worst_4_mae):
    print(f"    {n}: {m:.4f} {'OK' if m < 1.0 else 'BAD'}")
print(f"{'='*60}")
