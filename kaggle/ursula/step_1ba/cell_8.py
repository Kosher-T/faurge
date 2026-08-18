# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Train
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train only EQ head (trunk frozen)
# Phase 2: Unfreeze trunk, fine-tune everything (lower LR)

import torch.optim as optim
import json

# ── Phase 1: Train head only ─────────────────────────────────────────────────

# Freeze trunk
for param in model.trunk.parameters():
    param.requires_grad = False
print("Phase 1: Trunk frozen, training EQ head only")

optimizer = optim.Adam(model.head.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

best_test_loss = float('inf')
patience_counter = 0
train_losses = []
test_losses = []

print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
print(f"  Batch size: {BATCH_SIZE}")
print()

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        pred = model(batch_inputs)
        loss = criterion(pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_inputs)

    train_loss = epoch_loss / len(train_dataset)
    train_losses.append(train_loss)

    # ── Eval ──
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            pred = model(batch_inputs)
            loss = criterion(pred, batch_labels)
            test_loss += loss.item() * len(batch_inputs)

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

    # ── Logging ──
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d} | train: {train_loss:.6f} | test: {test_loss:.6f} | best: {best_test_loss:.6f} | lr: {lr:.2e}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
        break

print(f"\nPhase 1 done. Best test loss: {best_test_loss:.6f}")

# ── Save loss history ─────────────────────────────────────────────────────────

history = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_test_loss': best_test_loss,
    'epochs_trained': len(train_losses),
    'phase': 1,
}
with open(CHECKPOINT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"Model saved to: {MODEL_PATH}")
