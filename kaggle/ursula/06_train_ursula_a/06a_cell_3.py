# %% [markdown]
# ## Supervised Training
#
# Train the policy network to predict inverse degradation parameters
# from metric pairs. MSE regression on the 125D action vector.
# With thousands of meaningful targets (vs 3 from random search),
# the model learns physically grounded restoration parameters.
#
# Features: 80/20 train/test split, augmentation on train only,
# gradient clipping, weight decay, early stopping, loss curve logging.

import pickle

# ══════════════════════════════════════════════════════════════════════════════
# Load targets and do 80/20 Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════

with open(OUTPUT / 'supervised_targets.pkl', 'rb') as f:
    supervised_data = pickle.load(f)

# Convert to numpy arrays first
obs_np = np.array([d[0] for d in supervised_data])
tgt_np = np.array([d[1] for d in supervised_data])

# Shuffle and split indices 80/20
np.random.seed(42)  # Keep it reproducible
dataset_size = len(supervised_data)
indices = np.random.permutation(dataset_size)

train_size = int(0.8 * dataset_size)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

print(f"Total pairs: {dataset_size}")
print(f"Training pairs: {len(train_indices)} (80%)")
print(f"Testing pairs: {len(test_indices)} (20%)")

# Create Train and Test tensors
train_obs = torch.tensor(obs_np[train_indices], dtype=torch.float32, device=DEVICE)
train_tgt = torch.tensor(tgt_np[train_indices], dtype=torch.float32, device=DEVICE)

test_obs = torch.tensor(obs_np[test_indices], dtype=torch.float32, device=DEVICE)
test_tgt = torch.tensor(tgt_np[test_indices], dtype=torch.float32, device=DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# Augment ONLY the training data
# ══════════════════════════════════════════════════════════════════════════════

N_COPIES = max(1, SUPERVISED_EPOCHS // 10)
noise_std = AUGMENTATION_NOISE

# Repeat and add noise to training data
train_obs_aug = train_obs.repeat(N_COPIES, 1)
train_tgt_aug = train_tgt.repeat(N_COPIES, 1)
train_obs_aug += torch.randn_like(train_obs_aug) * noise_std

train_dataset = torch.utils.data.TensorDataset(train_obs_aug, train_tgt_aug)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=True)

print(f"\nAugmented training dataset: {len(train_dataset)} samples ({N_COPIES} copies × {len(train_indices)} pairs)")
print(f"Batches per epoch: {len(train_loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# Train policy (MSE regression on 125D output)
# ══════════════════════════════════════════════════════════════════════════════

policy = UrsulaPolicy().to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=SUPERVISED_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPERVISED_EPOCHS)

print(f"\n{'='*60}")
print(f"  SUPERVISED TRAINING")
print(f"{'='*60}")
print(f"  Model params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"  Epochs: {SUPERVISED_EPOCHS}")
print(f"  Batch size: {SUPERVISED_BATCH_SIZE}")
print(f"  LR: {SUPERVISED_LR}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Grad clip norm: {GRAD_CLIP_NORM}")
print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
print(f"{'='*60}\n")

# ── Loss curve logging ──
loss_history = {
    "epoch": [],
    "train_loss": [],
    "test_mse": [],
    "test_eq_mse": [],
    "test_gain_mse": [],
    "lr": [],
}

policy.train()
t_start = time.time()
best_test_mse = float('inf')
best_policy_state = None
epochs_without_improvement = 0

for epoch in range(1, SUPERVISED_EPOCHS + 1):
    epoch_loss = 0.0
    n_batches = 0

    for obs_batch, tgt_batch in train_loader:
        pred = policy(obs_batch)
        loss = F.mse_loss(pred, tgt_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)

    if epoch % 10 == 0 or epoch == 1:
        # Evaluate on the unseen TEST set
        policy.eval()
        with torch.no_grad():
            # Test set evaluation
            pred_test = policy(test_obs)
            test_mse = F.mse_loss(pred_test, test_tgt).item()

            # Deeper breakdown
            pred_np = pred_test.cpu().numpy()
            tgt_np_test = test_tgt.cpu().numpy()
            eq_mse = float(np.mean((pred_np[:, :124] - tgt_np_test[:, :124]) ** 2))
            gain_mse = float(np.mean((pred_np[:, 124:] - tgt_np_test[:, 124:]) ** 2))
        policy.train()

        # Log loss curve
        lr = scheduler.get_last_lr()[0]
        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(avg_loss)
        loss_history["test_mse"].append(test_mse)
        loss_history["test_eq_mse"].append(eq_mse)
        loss_history["test_gain_mse"].append(gain_mse)
        loss_history["lr"].append(lr)

        # Save best weights
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_policy_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            is_new_best = "⭐ NEW BEST"
            epochs_without_improvement = 0
        else:
            is_new_best = ""
            epochs_without_improvement += 10  # we only check every 10 epochs

        elapsed = time.time() - t_start
        print(f"  epoch {epoch:>4}/{SUPERVISED_EPOCHS} | train_loss {avg_loss:.6f} | TEST_MSE {test_mse:.6f} "
              f"{is_new_best} | EQ {eq_mse:.6f} Gain {gain_mse:.6f} "
              f"| lr {lr:.6f} | {elapsed:.0f}s")

        # Early stopping check
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n  ⛔ EARLY STOPPING at epoch {epoch} — no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

print(f"\n  Training complete — {time.time() - t_start:.0f}s")
print(f"  Best Test MSE: {best_test_mse:.6f}")

# Restore the best weights we found during training!
if best_policy_state is not None:
    policy.load_state_dict(best_policy_state)
    print("  Restored best model weights for saving.")

# Save loss curves for later analysis
loss_history_path = OUTPUT / 'loss_history.json'
with open(loss_history_path, 'w') as f:
    json.dump(loss_history, f, indent=2)
print(f"  Loss curves saved to {loss_history_path}")
