# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Train (50-frequency curriculum)
# ══════════════════════════════════════════════════════════════════════════════
# Filter precomputed data by frequency. No regeneration needed.

import torch.optim as optim
import json

# ── Load seed (Step 1ba best) ────────────────────────────────────────────────

load_with_expansion(model, STEP1BA_MODEL)
torch.save(model.state_dict(), CHECKPOINT_DIR / 'curriculum_seed.pt')
print(f"Loaded Step 1ba as seed")

# ── Precompute frequency index mapping ────────────────────────────────────────
# Each sample in all_inputs was generated at a specific frequency.
# We need to know which samples belong to which frequency.

freq_val_to_hz = {}
for freq_hz in FREQ_CURRICULUM:
    freq_val = freq_to_input(freq_hz)
    freq_val_to_hz[round(freq_val, 6)] = freq_hz

# Build index: for each frequency, which samples belong to it
freq_sample_indices = {}
for sample_idx in range(len(all_inputs)):
    freq_val = all_inputs[sample_idx, -1]  # last column is frequency
    key = round(float(freq_val), 6)
    if key not in freq_sample_indices:
        freq_sample_indices[key] = []
    freq_sample_indices[key].append(sample_idx)

print(f"Frequency index built: {len(freq_sample_indices)} unique frequencies")

# ── Sequential frequency training ─────────────────────────────────────────────

results = []

for step, freq_hz in enumerate(FREQ_CURRICULUM):
    print(f"\n{'='*60}")
    print(f"STEP {step+1}/{len(FREQ_CURRICULUM)}: {freq_hz} Hz")
    print(f"{'='*60}")

    # Filter samples for this frequency
    freq_val = freq_to_input(freq_hz)
    key = round(freq_val, 6)
    indices = np.array(freq_sample_indices.get(key, []))

    if len(indices) == 0:
        print(f"  WARNING: No samples for {freq_hz}Hz, skipping")
        continue

    freq_inputs = all_inputs[indices]
    freq_labels = all_labels[indices]

    # Split
    n = len(freq_inputs)
    idx = np.random.permutation(n)
    split = int(n * TRAIN_SPLIT)

    train_ds = EQGainDataset(freq_inputs[idx[:split]], freq_labels[idx[:split]])
    val_ds = EQGainDataset(freq_inputs[idx[split:]], freq_labels[idx[split:]])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Load previous best
    if step == 0:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / 'curriculum_seed.pt'))
    else:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / 'curriculum_best.pt'))

    optimizer = optim.Adam(model.parameters(), lr=LR)
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

    # Quick eval
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
print("SEQUENTIAL CURRICULUM RESULTS (135D: metrics + frequency)")
print(f"{'='*60}")
print(f"{'Step':>4} | {'Freq (Hz)':>10} | {'MAE (dB)':>8} | {'±1dB':>6} | {'Conv':>5}")
print(f"{'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")
for r in results:
    print(f"{r['step']:>4} | {r['freq_hz']:>10} | {r['mae']:>8.2f} | {r['within_1dB']:>5.0f}% | {r['converged_epoch']:>5}")

with open(CHECKPOINT_DIR / 'curriculum_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {CHECKPOINT_DIR / 'curriculum_results.json'}")
