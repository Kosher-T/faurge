# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Define Model (Load Step 1b, Replace Head)
# ══════════════════════════════════════════════════════════════════════════════
# Load Step 1b model (trunk + gain head), replace head for EQ gain prediction.
# Trunk already learned to process 134D input. Just teach new output.

import torch.nn as nn

class CausalityPredictor(nn.Module):
    """Same architecture as Step 1b, but with 1D output for EQ gain."""
    def __init__(self, input_dim=METRIC_DIM * 2, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h)

model = CausalityPredictor().to(device)

# ── Load Step 1b weights (trunk + old head) ──────────────────────────────────

if STEP1B_MODEL is not None:
    try:
        checkpoint = torch.load(STEP1B_MODEL, map_location=device)
        step1b_sd = checkpoint['model_state_dict']

        # Load trunk weights (same architecture)
        loadable = {}
        for k, v in step1b_sd.items():
            if k.startswith("trunk."):
                if k in model.state_dict():
                    if v.shape == model.state_dict()[k].shape:
                        loadable[k] = v

        model.load_state_dict(loadable, strict=False)
        loaded = len(loadable)
        print(f"Loaded {loaded} trunk weights from Step 1b")
    except Exception as e:
        print(f"Could not load Step 1b weights: {e}")
        print("Starting with fresh weights")

# ── Summary ───────────────────────────────────────────────────────────────────

n_trunk = sum(p.numel() for p in model.trunk.parameters())
n_head = sum(p.numel() for p in model.head.parameters())
print(f"\nTrunk: {n_trunk:,} parameters (loaded from Step 1b)")
print(f"EQ Head: {n_head:,} parameters (fresh, will train)")
print(f"Input:  {METRIC_DIM * 2}D (degraded + clean metrics)")
print(f"Output: {OUTPUT_DIM}D (gain_db only)")
print(model)
