# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Build DataLoaders
# ══════════════════════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset, DataLoader

class EQGainDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ── Train/test split ─────────────────────────────────────────────────────────

n = len(all_inputs)
idx = np.random.permutation(n)
split = int(n * TRAIN_SPLIT)

train_dataset = EQGainDataset(all_inputs[idx[:split]], all_labels[idx[:split]])
test_dataset = EQGainDataset(all_inputs[idx[split:]], all_labels[idx[split:]])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Test:  {len(test_dataset)} samples")

batch_x, batch_y = next(iter(train_loader))
print(f"\nBatch shapes: inputs={batch_x.shape}, labels={batch_y.shape}")
print(f"Input range:  [{batch_x.min():.2f}, {batch_x.max():.2f}]")
print(f"Label range:  [{batch_y.min():.2f}, {batch_y.max():.2f}]")
