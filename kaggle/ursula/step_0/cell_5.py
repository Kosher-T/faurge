# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Build Dataset
# ══════════════════════════════════════════════════════════════════════════════
# Pair degraded metrics with clean metrics. Train/test split. DataLoader.

import torch
from torch.utils.data import Dataset, DataLoader

# ── Dataset ───────────────────────────────────────────────────────────────────

class MetricDataset(Dataset):
    def __init__(self, degraded, clean):
        self.degraded = torch.tensor(degraded, dtype=torch.float32)
        self.clean = torch.tensor(clean, dtype=torch.float32)

    def __len__(self):
        return len(self.degraded)

    def __getitem__(self, idx):
        return self.degraded[idx], self.clean[idx]

# ── Pair degraded with clean metrics ─────────────────────────────────────────

# For each degraded sample, pair it with its clip's clean metrics
paired_clean = all_clean_metrics[all_clip_ids]  # (N, 67) — broadcast clean to match degraded

# ── Train/test split ─────────────────────────────────────────────────────────

n_total = len(all_degraded_metrics)
n_train = int(n_total * TRAIN_SPLIT)

# Shuffle indices
indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

train_degraded = all_degraded_metrics[train_idx]
train_clean = paired_clean[train_idx]
test_degraded = all_degraded_metrics[test_idx]
test_clean = paired_clean[test_idx]

print(f"Train: {len(train_degraded)} pairs")
print(f"Test:  {len(test_degraded)} pairs")

# ── DataLoaders ───────────────────────────────────────────────────────────────

train_dataset = MetricDataset(train_degraded, train_clean)
test_dataset = MetricDataset(test_degraded, test_clean)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Sanity check
batch_degraded, batch_clean = next(iter(train_loader))
print(f"\nBatch shapes: degraded={batch_degraded.shape}, clean={batch_clean.shape}")
print(f"Degraded range: [{batch_degraded.min():.1f}, {batch_degraded.max():.1f}]")
print(f"Clean range:    [{batch_clean.min():.1f}, {batch_clean.max():.1f}]")
