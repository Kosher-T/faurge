# %% [markdown]
# ## Supervised Training
#
# Train the policy network to predict inverse degradation parameters
# from metric pairs. MSE regression on the 125D action vector.
# With thousands of meaningful targets (vs 3 from random search),
# the model learns physically grounded restoration parameters.

import pickle

# ══════════════════════════════════════════════════════════════════════════════
# Load targets
# ══════════════════════════════════════════════════════════════════════════════

with open(OUTPUT / 'supervised_targets.pkl', 'rb') as f:
    supervised_data = pickle.load(f)

# Convert to tensors
observations = torch.tensor(np.array([d[0] for d in supervised_data]), dtype=torch.float32, device=DEVICE)
targets = torch.tensor(np.array([d[1] for d in supervised_data]), dtype=torch.float32, device=DEVICE)
mses = np.array([d[2] for d in supervised_data])

print(f"Loaded {len(supervised_data)} supervised pairs")
print(f"  Observations: {observations.shape}")
print(f"  Targets: {targets.shape}")
print(f"  MSE range: [{mses.min():.2f}, {mses.max():.2f}]")

# ══════════════════════════════════════════════════════════════════════════════
# Augment data: replicate with noise to improve generalization
# ══════════════════════════════════════════════════════════════════════════════

N_COPIES = max(1, SUPERVISED_EPOCHS // 10)  # fewer copies, more real pairs
noise_std = AUGMENTATION_NOISE

obs_aug = observations.repeat(N_COPIES, 1)
tgt_aug = targets.repeat(N_COPIES, 1)
obs_aug += torch.randn_like(obs_aug) * noise_std

dataset = torch.utils.data.TensorDataset(obs_aug, tgt_aug)
loader = torch.utils.data.DataLoader(dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=True)

print(f"\nAugmented dataset: {len(dataset)} samples ({N_COPIES} copies × {len(supervised_data)} pairs)")
print(f"Batches per epoch: {len(loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# Train policy (MSE regression on 125D output)
# ══════════════════════════════════════════════════════════════════════════════

policy = UrsulaPolicy().to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=SUPERVISED_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPERVISED_EPOCHS)

print(f"\n{'='*60}")
print(f"  SUPERVISED TRAINING")
print(f"{'='*60}")
print(f"  Model params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"  Epochs: {SUPERVISED_EPOCHS}")
print(f"  Batch size: {SUPERVISED_BATCH_SIZE}")
print(f"  LR: {SUPERVISED_LR}")
print(f"{'='*60}\n")

policy.train()
t_start = time.time()
best_eval_mse = float('inf')

for epoch in range(1, SUPERVISED_EPOCHS + 1):
    epoch_loss = 0.0
    n_batches = 0

    for obs_batch, tgt_batch in loader:
        pred = policy(obs_batch)
        loss = F.mse_loss(pred, tgt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)

    if epoch % 10 == 0 or epoch == 1:
        # Evaluate on original (non-augmented) data
        policy.eval()
        with torch.no_grad():
            pred_all = policy(observations)
            eval_mse = F.mse_loss(pred_all, targets).item()
            pred_np = pred_all.cpu().numpy()
            tgt_np = targets.cpu().numpy()
            eq_mse = np.mean((pred_np[:, :124] - tgt_np[:, :124]) ** 2)
            gain_mse = np.mean((pred_np[:, 124:] - tgt_np[:, 124:]) ** 2)
        policy.train()

        if eval_mse < best_eval_mse:
            best_eval_mse = eval_mse

        elapsed = time.time() - t_start
        lr = scheduler.get_last_lr()[0]
        print(f"  epoch {epoch:>4}/{SUPERVISED_EPOCHS} | loss {avg_loss:.6f} | eval_mse {eval_mse:.6f} "
              f"(best {best_eval_mse:.6f}) | EQ {eq_mse:.6f} Gain {gain_mse:.6f} "
              f"| lr {lr:.6f} | {elapsed:.0f}s")

print(f"\n  Training complete — {time.time() - t_start:.0f}s")
print(f"  Best eval MSE: {best_eval_mse:.6f}")
