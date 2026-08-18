# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Build Dataset
# ══════════════════════════════════════════════════════════════════════════════
# Input:  degraded metrics (67D) + clean metrics (67D) = 134D
# Output: plugin parameter(s) (OUTPUT_DIM)

import torch
from torch.utils.data import Dataset, DataLoader

# ── Dataset ───────────────────────────────────────────────────────────────────

class CausalityDataset(Dataset):
    def __init__(self, degraded_metrics, clean_metrics, labels):
        """
        Args:
            degraded_metrics: (N, 67) degraded metric vectors
            clean_metrics: (N, 67) clean metric vectors
            labels: (N, OUTPUT_DIM) plugin parameter vectors
        """
        self.inputs = torch.tensor(
            np.concatenate([degraded_metrics, clean_metrics], axis=1),
            dtype=torch.float32
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ── Pair degraded with clean metrics ─────────────────────────────────────────

paired_clean = all_clean_metrics[all_clip_ids]

# ── Train/test split ─────────────────────────────────────────────────────────

n_total = len(all_degraded_metrics)
n_train = int(n_total * TRAIN_SPLIT)

indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

train_degraded = all_degraded_metrics[train_idx]
train_clean = paired_clean[train_idx]
train_labels = all_labels[train_idx]

test_degraded = all_degraded_metrics[test_idx]
test_clean = paired_clean[test_idx]
test_labels = all_labels[test_idx]

print(f"Train: {len(train_degraded)} pairs")
print(f"Test:  {len(test_degraded)} pairs")

# ── DataLoaders ───────────────────────────────────────────────────────────────

train_dataset = CausalityDataset(train_degraded, train_clean, train_labels)
test_dataset = CausalityDataset(test_degraded, test_clean, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Sanity check
batch_inputs, batch_labels = next(iter(train_loader))
print(f"\nBatch shapes: inputs={batch_inputs.shape}, labels={batch_labels.shape}")
print(f"Input range:  [{batch_inputs.min():.2f}, {batch_inputs.max():.2f}]")
print(f"Label range:  [{batch_labels.min():.2f}, {batch_labels.max():.2f}]")
