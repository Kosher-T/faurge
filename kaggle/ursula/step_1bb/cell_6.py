# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Define Model (135D → 1D, load Step 1ba trunk)
# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba trunk: 134D → 128 → 128
# Expand first layer: 134 → 135 (add frequency input)
# Load Step 1ba weights, zero-init new frequency column

import torch.nn as nn

class EQGainPredictor(nn.Module):
    """135D input (134 metrics + 1 frequency) → 1D gain output."""
    def __init__(self, input_dim=METRIC_DIM * 2 + 1, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
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

model = EQGainPredictor().to(device)

# ── Helper: load weights with 134→135 expansion ─────────────────────────────

def load_with_expansion(model, checkpoint_path):
    """Load 134D Step 1ba weights into 135D model. Zero-init frequency column."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    old_sd = checkpoint.get('model_state_dict', checkpoint)
    new_sd = model.state_dict()
    loadable = {}

    for k, v in old_sd.items():
        if k not in new_sd:
            continue
        if v.shape == new_sd[k].shape:
            loadable[k] = v
        elif k == 'trunk.0.weight':
            # LayerNorm: expand 134 → 135
            new_v = torch.zeros(135, device=device)
            new_v[:134] = v
            loadable[k] = new_v
            print(f"  Expanded {k}: {list(v.shape)} → {list(new_v.shape)}")
        elif k == 'trunk.0.bias':
            # LayerNorm: expand 134 → 135
            new_v = torch.zeros(135, device=device)
            new_v[:134] = v
            loadable[k] = new_v
            print(f"  Expanded {k}: {list(v.shape)} → {list(new_v.shape)}")
        elif k == 'trunk.1.weight':
            # Linear: expand [128, 134] → [128, 135]
            new_v = torch.zeros(128, 135, device=device)
            new_v[:, :134] = v
            loadable[k] = new_v
            print(f"  Expanded {k}: {list(v.shape)} → {list(new_v.shape)}")

    model.load_state_dict(loadable, strict=False)
    return len(loadable)

# ── Load Step 1ba ─────────────────────────────────────────────────────────────

n_loaded = load_with_expansion(model, STEP1BA_MODEL)
print(f"Loaded {n_loaded} weights from Step 1ba")

# ── Summary ───────────────────────────────────────────────────────────────────

n_trunk = sum(p.numel() for p in model.trunk.parameters())
n_head = sum(p.numel() for p in model.head.parameters())
print(f"\nTrunk: {n_trunk:,} parameters")
print(f"Head:  {n_head:,} parameters")
print(f"Input:  {METRIC_DIM * 2 + 1}D (degraded + clean metrics + frequency)")
print(f"Output: {OUTPUT_DIM}D (gain_db)")
print(model)
