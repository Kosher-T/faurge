# ══════════════════════════════════════════════════════════════════════════════
# Step 1bc — Train (160D → 2D, frequency + gain)
# ══════════════════════════════════════════════════════════════════════════════

import torch.optim as optim
import json

# ── Weighted loss ─────────────────────────────────────────────────────────────

class FreqGainLoss(nn.Module):
    """MSE loss with separate weights for frequency and gain."""
    def __init__(self, freq_weight=FREQ_LOSS_WEIGHT):
        super().__init__()
        self.freq_weight = freq_weight
        self.gain_weight = 1.0 - freq_weight

    def forward(self, pred, target):
        freq_loss = nn.functional.mse_loss(pred[:, 0], target[:, 0])
        gain_loss = nn.functional.mse_loss(pred[:, 1], target[:, 1])
        return self.freq_weight * freq_loss + self.gain_weight * gain_loss, freq_loss, gain_loss

criterion = FreqGainLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

best_test_loss = float('inf')
patience_counter = 0
train_losses = []
test_losses = []

print(f"Training: {EPOCHS} epochs, patience={PATIENCE}, LR={LR}, batch={BATCH_SIZE}")
print(f"Loss weights: freq={FREQ_LOSS_WEIGHT}, gain={1-FREQ_LOSS_WEIGHT}")
print()

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    epoch_loss = 0.0
    epoch_freq = 0.0
    epoch_gain = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss, freq_loss, gain_loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_x)
        epoch_freq += freq_loss.item() * len(batch_x)
        epoch_gain += gain_loss.item() * len(batch_x)

    train_loss = epoch_loss / len(train_dataset)
    train_freq = epoch_freq / len(train_dataset)
    train_gain = epoch_gain / len(train_dataset)
    train_losses.append(train_loss)

    # ── Eval ──
    model.eval()
    test_loss = 0.0
    test_freq = 0.0
    test_gain = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss, freq_loss, gain_loss = criterion(pred, batch_y)
            test_loss += loss.item() * len(batch_x)
            test_freq += freq_loss.item() * len(batch_x)
            test_gain += gain_loss.item() * len(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_dataset)
    test_freq /= len(test_dataset)
    test_gain /= len(test_dataset)
    test_losses.append(test_loss)

    # ── Compute accuracy metrics ──
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    pred_freq_hz = denormalize_freq(preds[:, 0])
    true_freq_hz = denormalize_freq(targets[:, 0])
    pred_gain_db = denormalize_gain(preds[:, 1])
    true_gain_db = denormalize_gain(targets[:, 1])

    freq_mae_hz = np.mean(np.abs(pred_freq_hz - true_freq_hz))
    freq_mae_pct = np.mean(np.abs(pred_freq_hz - true_freq_hz) / (true_freq_hz + 1e-6)) * 100
    gain_mae = np.mean(np.abs(pred_gain_db - true_gain_db))
    gain_within_1 = np.mean(np.abs(pred_gain_db - true_gain_db) <= 1.0) * 100

    scheduler.step(test_loss)

    # ── Checkpoint ──
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_loss': test_loss,
            'freq_mae_hz': float(freq_mae_hz),
            'gain_mae': float(gain_mae),
        }, MODEL_PATH)
    else:
        patience_counter += 1

    lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | loss: {test_loss:.4f} (freq:{test_freq:.4f} gain:{test_gain:.4f}) | "
              f"freq_MAE: {freq_mae_hz:.0f}Hz ({freq_mae_pct:.1f}%) | gain_MAE: {gain_mae:.2f}dB ±1dB:{gain_within_1:.0f}% | lr: {lr:.1e}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
        break

print(f"\nBest test loss: {best_test_loss:.6f}")
print(f"Model saved to: {MODEL_PATH}")

# ── Save history ──────────────────────────────────────────────────────────────

history = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_test_loss': best_test_loss,
    'epochs_trained': len(train_losses),
}
with open(CHECKPOINT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)
