# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Define Model
# ══════════════════════════════════════════════════════════════════════════════
# MLP: 134D (degraded + clean metrics) → hidden → OUTPUT_DIM (plugin params)
# Loads Step 0 trunk weights if LOAD_MODEL_FROM is set.

import torch.nn as nn

class CausalityPredictor(nn.Module):
    def __init__(self, input_dim=METRIC_DIM * 2, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Trunk (same structure as Step 0, but input is 134D)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Output head (variable dim based on active plugins)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h)

model = CausalityPredictor().to(device)

# ── Load Step 0 trunk weights (if available) ─────────────────────────────────

if LOAD_MODEL_FROM is not None:
    try:
        checkpoint = torch.load(LOAD_MODEL_FROM, map_location=device)
        step0_sd = checkpoint['model_state_dict']

        # Map Step 0 trunk weights to Step 1 trunk
        # Step 0: net.0 (LayerNorm), net.1 (Linear), net.3 (Linear), net.5 (Linear)
        # Step 1: trunk.0 (LayerNorm), trunk.1 (Linear), trunk.3 (Linear), trunk.5 (Linear)
        loadable = {}
        for k, v in step0_sd.items():
            if k.startswith("net."):
                new_key = "trunk." + k[4:]  # "net." → "trunk."
                if new_key in model.state_dict():
                    if v.shape == model.state_dict()[new_key].shape:
                        loadable[new_key] = v

        model.load_state_dict(loadable, strict=False)
        loaded = len(loadable)
        print(f"Loaded {loaded} weights from Step 0 trunk")
    except Exception as e:
        print(f"Could not load Step 0 weights: {e}")
        print("Starting with fresh weights")

# ── Summary ───────────────────────────────────────────────────────────────────

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {n_params:,} parameters")
print(f"Input:  {METRIC_DIM * 2}D (degraded + clean metrics)")
print(f"Output: {OUTPUT_DIM}D (plugin parameters)")
print(model)
