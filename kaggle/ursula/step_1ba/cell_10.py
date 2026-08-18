# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Sequential Frequency Curriculum (24 Frequencies)
# ══════════════════════════════════════════════════════════════════════════════
# Train on one frequency, save best, load for next, repeat.
# Starting point: Step 1b trunk + fresh head.
# Tests: does the model learn a general EQ concept as it sees more frequencies?

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

# ── 24 frequencies, logarithmically spaced across the spectrum ────────────────

FREQ_CURRICULUM = [
    100, 150, 220, 330, 470,      # Low: 100-470Hz
    680, 1000, 1500, 2200, 3300,   # Mid: 680-3300Hz
    4700, 6800, 10000, 15000,      # High: 4.7k-15kHz
    200, 400, 800, 1600, 3200,     # Second pass: fill gaps
    600, 1200, 2400, 4800, 9600,   # More coverage
]

N_DEGRADATIONS = 500
FREQ_EPOCHS = 300
FREQ_PATIENCE = 50

# ── Load Step 1b trunk as starting point ──────────────────────────────────────

STARTER_MODEL = "/kaggle/input/notebooks/itorousa/ur-step-1/step_1/checkpoints/best_model.pt"

model = CausalityPredictor(output_dim=1).to(device)
try:
    checkpoint = torch.load(STARTER_MODEL, map_location=device)
    step1b_sd = checkpoint['model_state_dict']
    loadable = {}
    for k, v in step1b_sd.items():
        if k.startswith("trunk."):
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
                loadable[k] = v
    model.load_state_dict(loadable, strict=False)
    print(f"Loaded {len(loadable)} trunk weights from Step 1b")
except Exception as e:
    print(f"Could not load Step 1b: {e} — starting fresh")

# Save this as the "seed" model
torch.save(model.state_dict(), CHECKPOINT_DIR / 'curriculum_seed.pt')

# ── Run sequential curriculum ─────────────────────────────────────────────────

results = []

for step, freq_hz in enumerate(FREQ_CURRICULUM):
    print(f"\n{'='*60}")
    print(f"STEP {step+1}/{len(FREQ_CURRICULUM)}: {freq_hz} Hz")
    print(f"{'='*60}")

    # ── Generate data for this frequency ────────────────────────────────────
    all_clean = []
    all_degraded = []
    all_labels = []

    for clip in clean_clips:
        clean_m = extract_metrics_67d(clip)
        for _ in range(N_DEGRADATIONS):
            gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
            degraded, _ = eq_plugin.process(
                clip, sr=SR,
                bands=[{"freq_hz": float(freq_hz), "gain_db": float(gain_db),
                        "q": EQ_FIXED_Q, "filter_type": "peak"}]
            )
            degraded = np.clip(degraded, -1.0, 1.0)
            deg_m = extract_metrics_67d(degraded)
            all_clean.append(clean_m)
            all_degraded.append(deg_m)
            all_labels.append([gain_db])

    all_clean = np.array(all_clean)
    all_degraded = np.array(all_degraded)
    all_labels = np.array(all_labels)

    # ── Build loaders ────────────────────────────────────────────────────────
    inputs = np.concatenate([all_degraded, all_clean], axis=1)
    n = len(inputs)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    train_x = torch.tensor(inputs[idx[:split]], dtype=torch.float32)
    train_y = torch.tensor(all_labels[idx[:split]], dtype=torch.float32)
    val_x = torch.tensor(inputs[idx[split:]], dtype=torch.float32)
    val_y = torch.tensor(all_labels[idx[split:]], dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    val_ds = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

    # ── Train (load previous best) ───────────────────────────────────────────
    if step == 0:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / 'curriculum_seed.pt'))
    else:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / 'curriculum_best.pt'))
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

    best_loss = float('inf')
    patience_ct = 0
    converged_epoch = None

    for epoch in range(FREQ_EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(bx)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_ct = 0
            converged_epoch = epoch + 1
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'curriculum_best.pt')
        else:
            patience_ct += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | val: {val_loss:.4f} | best: {best_loss:.4f}")

        if patience_ct >= FREQ_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    # ── Quick eval on this frequency ─────────────────────────────────────────
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            pred = model(bx)
            preds.append(pred.cpu().numpy())
            targets.append(by.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = np.mean(np.abs(preds - targets))
    within_1 = np.mean(np.abs(preds - targets) <= 1.0) * 100

    results.append({
        'step': step + 1,
        'freq_hz': freq_hz,
        'mae': float(mae),
        'within_1dB': float(within_1),
        'best_loss': float(best_loss),
        'converged_epoch': converged_epoch,
    })
    print(f"  RESULT: {freq_hz}Hz → MAE={mae:.2f}dB, ±1dB={within_1:.0f}%, converged@{converged_epoch}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SEQUENTIAL CURRICULUM RESULTS")
print(f"{'='*60}")
print(f"{'Step':>4} | {'Freq (Hz)':>10} | {'MAE (dB)':>8} | {'±1dB':>6} | {'Conv':>5}")
print(f"{'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")
for r in results:
    print(f"{r['step']:>4} | {r['freq_hz']:>10} | {r['mae']:>8.2f} | {r['within_1dB']:>5.0f}% | {r['converged_epoch']:>5}")

# ── Save ──────────────────────────────────────────────────────────────────────

with open(CHECKPOINT_DIR / 'curriculum_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {CHECKPOINT_DIR / 'curriculum_results.json'}")
