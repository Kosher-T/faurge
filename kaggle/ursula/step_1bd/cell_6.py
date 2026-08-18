# ══════════════════════════════════════════════════════════════════════════════
# Step 1bd — DataLoaders (FREQUENCY-AWARE SPLIT)
# ══════════════════════════════════════════════════════════════════════════════
# Split by frequency: hold out 27 frequencies for unseen test.
# Train on 120 frequencies, evaluate on both seen and unseen.

import torch
from torch.utils.data import Dataset, DataLoader

class EQDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ── Frequency-aware split ────────────────────────────────────────────────────
# Hold out every 6th frequency for unseen test (~27 frequencies)

unique_freqs = np.unique(all_freq_hz)
unseen_freq_set = set(unique_freqs[::6])  # every 6th = ~27 frequencies
train_freq_set = set(unique_freqs) - unseen_freq_set

# Split samples by frequency
train_mask = np.array([f in train_freq_set for f in all_freq_hz])
unseen_mask = np.array([f in unseen_freq_set for f in all_freq_hz])

# For training frequencies, do 80/20 random split
train_indices = np.where(train_mask)[0]
np.random.seed(42)
np.random.shuffle(train_indices)
split = int(len(train_indices) * TRAIN_SPLIT)

train_idx = train_indices[:split]
seen_test_idx = train_indices[split:]
unseen_test_idx = np.where(unseen_mask)[0]

print(f"Unique frequencies: {len(unique_freqs)}")
print(f"  Seen (train):     {len(train_freq_set)} frequencies")
print(f"  Unseen (test):    {len(unseen_freq_set)} frequencies")
print(f"  Unseen freqs:     {sorted(unseen_freq_set)[:10]}...")

# ── Create datasets ──────────────────────────────────────────────────────────

train_dataset = EQDataset(all_inputs[train_idx], all_labels[train_idx])
seen_test_dataset = EQDataset(all_inputs[seen_test_idx], all_labels[seen_test_idx])
unseen_test_dataset = EQDataset(all_inputs[unseen_test_idx], all_labels[unseen_test_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
seen_test_loader = DataLoader(seen_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
unseen_test_loader = DataLoader(unseen_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# For training, combine train + seen_test as the full "training distribution" test set
# (used during training for early stopping)
test_dataset = seen_test_dataset
test_loader = seen_test_loader

print(f"\nTrain:      {len(train_dataset):,} samples ({len(train_freq_set)} freqs)")
print(f"Seen test:  {len(seen_test_dataset):,} samples ({len(train_freq_set)} freqs)")
print(f"Unseen test:{len(unseen_test_dataset):,} samples ({len(unseen_freq_set)} freqs)")

batch_x, batch_y = next(iter(train_loader))
print(f"\nBatch shapes: inputs={batch_x.shape}, labels={batch_y.shape}")
print(f"Input range:  [{batch_x.min():.2f}, {batch_x.max():.2f}]")
print(f"Label range:  freq=[{batch_y[:,0].min():.3f}, {batch_y[:,0].max():.3f}], gain=[{batch_y[:,1].min():.3f}, {batch_y[:,1].max():.3f}]")
