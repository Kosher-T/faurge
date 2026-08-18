# ══════════════════════════════════════════════════════════════════════════════
# Step 1bd — Model (160D → 2D)
# ══════════════════════════════════════════════════════════════════════════════
# Input:  160D (80 degraded + 80 clean metrics)
# Output: 2D (normalized log_freq, normalized gain)
# Load Step 1bc trunk (80D→128→128), expand to 160D, add new head.

import torch.nn as nn

class EQPredictor(nn.Module):
    """160D input → 2D output (frequency + gain)."""
    def __init__(self, input_dim=METRIC_DIM * 2, hidden_dim=HIDDEN_DIM, output_dim=2):
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

model = EQPredictor().to(device)

# ── Load Step 1bc trunk (80D → 160D) ───────────────────────────────────────
# Step 1bc: net.0=LayerNorm(80), net.1=Linear(80,128), net.4=Linear(128,128)
# Step 1bd: trunk.0=LayerNorm(160), trunk.1=Linear(160,128), trunk.4=Linear(128,128)
# Expand degraded side [0:80] → [0:80], clean side [0:80] → [80:160]

def load_step1bc_trunk(model, checkpoint_path):
    """Load Step 1bc metric literacy weights, expand 80D → 160D."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    old_sd = checkpoint.get('model_state_dict', checkpoint)
    new_sd = model.state_dict()
    loadable = {}

    for k, v in old_sd.items():
        # Map net.X → trunk.X
        new_key = k.replace('net.', 'trunk.')
        if new_key not in new_sd:
            continue

        if v.shape == new_sd[new_key].shape:
            loadable[new_key] = v
        elif 'trunk.0' in new_key and 'weight' in new_key:
            # LayerNorm(80) → LayerNorm(160): degraded [0:80] → [0:80], clean [0:80] → [80:160]
            new_v = torch.zeros(160, device=device)
            new_v[:80] = v
            new_v[80:160] = v
            loadable[new_key] = new_v
            print(f"  Expanded {new_key}: {list(v.shape)} → {list(new_v.shape)}")
        elif 'trunk.0' in new_key and 'bias' in new_key:
            new_v = torch.zeros(160, device=device)
            new_v[:80] = v
            new_v[80:160] = v
            loadable[new_key] = new_v
            print(f"  Expanded {new_key}: {list(v.shape)} → {list(new_v.shape)}")
        elif 'trunk.1' in new_key and 'weight' in new_key:
            # Linear(80,128) → Linear(160,128): duplicate columns
            new_v = torch.zeros(128, 160, device=device)
            new_v[:, :80] = v
            new_v[:, 80:160] = v
            loadable[new_key] = new_v
            print(f"  Expanded {new_key}: {list(v.shape)} → {list(new_v.shape)}")

    model.load_state_dict(loadable, strict=False)
    return len(loadable)

n_loaded = load_step1bc_trunk(model, STEP1BC_MODEL)
print(f"\nLoaded {n_loaded} trunk weights from Step 1bc (80D → 160D)")
print("Degraded and clean sides share weights initially")

# ── Summary ───────────────────────────────────────────────────────────────────

n_trunk = sum(p.numel() for p in model.trunk.parameters())
n_head = sum(p.numel() for p in model.head.parameters())
print(f"\nTrunk: {n_trunk:,} parameters (from Step 1bc)")
print(f"Head:  {n_head:,} parameters (fresh)")
print(f"Input:  {METRIC_DIM * 2}D (80 degraded + 80 clean metrics)")
print(f"Output: 2D (norm_log_freq, norm_gain)")
print(model)
