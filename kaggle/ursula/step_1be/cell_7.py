# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Phase 2: EQ Prediction Model (162D → 3D)
# ══════════════════════════════════════════════════════════════════════════════
# Load Phase 1 encoder, build EQ prediction head.
# Input:  162D = [degraded_81D, reference_81D]
# Output: 3D = [freq_norm, eq_gain_norm, global_gain_norm]
# Finetune: encoder unfreezes during training.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Load Phase 1 encoder ────────────────────────────────────────────────────
class EQPredictionNet(nn.Module):
    """162D input → 3D output (freq, eq_gain, global_gain).
    Initialized from Phase 1 encoder weights."""
    def __init__(self, input_dim=METRIC_DIM * 2, hidden=HIDDEN_DIM, output_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

# Initialize from Phase 1 encoder
eq_model = EQPredictionNet().to(device)

# Load Phase 1 encoder weights
phase1_enc_state = torch.load(PHASE1_ENCODER_PATH, map_location=device)
eq_model.encoder.load_state_dict(phase1_enc_state)
print("Loaded Phase 1 encoder weights into EQ model")

n_params = sum(p.numel() for p in eq_model.parameters())
print(f"EQ Model: {n_params:,} parameters")
print(f"Input:  {METRIC_DIM * 2}D | Output: 3D (freq, eq_gain, global_gain)")
