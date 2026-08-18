# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Define Model
# ══════════════════════════════════════════════════════════════════════════════
# Simple MLP: 67D degraded → 128 → 128 → 67D clean

import torch.nn as nn

class MetricPredictor(nn.Module):
    def __init__(self, input_dim=METRIC_DIM, hidden_dim=HIDDEN_DIM, output_dim=METRIC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MetricPredictor().to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_params:,} parameters")
print(f"Input:  {METRIC_DIM}D (degraded metrics)")
print(f"Output: {METRIC_DIM}D (predicted clean metrics)")
print(model)
