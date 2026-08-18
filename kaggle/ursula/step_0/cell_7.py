# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Train
# ══════════════════════════════════════════════════════════════════════════════
# Step 0:  Fresh training (LOAD_MODEL_FROM = None)
# Step 0b: Fine-tune from Step 0 checkpoint (LOAD_MODEL_FROM = checkpoint path)

import torch.optim as optim
import json

# ── Load pretrained model (step 0b) or start fresh (step 0) ──────────────────

if LOAD_MODEL_FROM is not None:
    print(f"Loading pretrained model from: {LOAD_MODEL_FROM}")
    checkpoint = torch.load(LOAD_MODEL_FROM, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']+1}, test_loss: {checkpoint['test_loss']:.6f}")
else:
    print("Starting fresh training (no pretrained model)")

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────

best_test_loss = float('inf')
patience_counter = 0
train_losses = []
test_losses = []

print("Training...")
print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
print(f"  Batch size: {BATCH_SIZE}")
print()

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    epoch_loss = 0.0
    for batch_degraded, batch_clean in train_loader:
        batch_degraded = batch_degraded.to(device)
        batch_clean = batch_clean.to(device)

        pred = model(batch_degraded)
        loss = criterion(pred, batch_clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_degraded)

    train_loss = epoch_loss / len(train_dataset)
    train_losses.append(train_loss)

    # ── Eval ──
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_degraded, batch_clean in test_loader:
            batch_degraded = batch_degraded.to(device)
            batch_clean = batch_clean.to(device)
            pred = model(batch_degraded)
            loss = criterion(pred, batch_clean)
            test_loss += loss.item() * len(batch_degraded)

    test_loss = test_loss / len(test_dataset)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    # ── Checkpoint ──
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_loss': test_loss,
            'train_loss': train_loss,
        }, MODEL_PATH)
    else:
        patience_counter += 1

    # ── Logging (every epoch — only 30 total) ──
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d} | train: {train_loss:.6f} | test: {test_loss:.6f} | best: {best_test_loss:.6f} | lr: {lr:.2e}")

    # ── Early stopping ──
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
        break

# ── Save loss history ─────────────────────────────────────────────────────────

history = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_test_loss': best_test_loss,
    'epochs_trained': len(train_losses),
}
with open(CHECKPOINT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\nDone. Best test loss: {best_test_loss:.6f}")
print(f"Model saved to: {MODEL_PATH}")
