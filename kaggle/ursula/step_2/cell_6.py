# ══════════════════════════════════════════════════════════════════════════════
# Step 1be — DataLoaders + Model + Train + Evaluate
# ══════════════════════════════════════════════════════════════════════════════
# All-in-one: loaders, 81D→81D MLP, train, evaluate.
# Data is pre-normalized per-dim to [0,1]. Evaluation denormalizes for MAE.

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
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} parameters")
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

# Denormalize to original scale
preds = denormalize(preds_norm)
targets = denormalize(targets_norm)

mae = np.mean(np.abs(preds - targets))
per_dim_mae = np.mean(np.abs(preds - targets), axis=0)

print(f"\n{'='*60}")
print(f"STEP 1be EVALUATION — 81D Metric Literacy (normalized training)")
print(f"{'='*60}")
print(f"Overall MAE: {mae:.6f}")

dim_names = (
    [f'LTAS_{i}' for i in range(64)] +
    ['LUFS', 'Crest', 'ZCR'] +
    ['Centroid', 'Bandwidth', 'Flatness', 'Flux', 'Rolloff', 'Skewness', 'Kurtosis', 'Slope'] +
    ['Sub', 'LowMid', 'Mid', 'Presence', 'Air'] +
    ['RMS']
)

print(f"\nPer-dimension MAE (top 10 worst):")
worst_idx = np.argsort(per_dim_mae)[::-1][:10]
for i in worst_idx:
    print(f"  {dim_names[i]:>12}: {per_dim_mae[i]:.4f}")

print(f"\nPer-dimension MAE (best 5):")
best_idx = np.argsort(per_dim_mae)[:5]
for i in best_idx:
    print(f"  {dim_names[i]:>12}: {per_dim_mae[i]:.6f}")

# Tier breakdown
tier0_mae = np.mean(per_dim_mae[:67])
tier1_mae = np.mean(per_dim_mae[67:80])
rms_mae = per_dim_mae[80]
print(f"\nTier 0 (67D) MAE: {tier0_mae:.4f}")
print(f"Tier 1 (13D) MAE: {tier1_mae:.4f}")
print(f"RMS (1D) MAE:     {rms_mae:.4f}")

print(f"\n{'='*60}")
if tier1_mae < 5.0 and rms_mae < 3.0:
    print("✅ PASSED — model learns all 81D metrics including RMS")
elif tier1_mae < 20.0:
    print("⚠️  MARGINAL — Tier 1 learning but weak")
else:
    print("❌ FAILED — Tier 1 features not learned")
print(f"{'='*60}")
