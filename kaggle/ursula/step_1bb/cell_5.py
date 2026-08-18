# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Generate or Load Dataset
# ══════════════════════════════════════════════════════════════════════════════
# If CACHE_PATH is set, load from disk. Otherwise generate and save.

import os

if CACHE_PATH and os.path.exists(CACHE_PATH):
    # ── Load from cache ──────────────────────────────────────────────────────
    print(f"Loading cached data from: {CACHE_PATH}")
    data = np.load(CACHE_PATH)
    all_inputs = data['inputs']
    all_labels = data['labels']
    all_clip_ids = data['clip_ids']
    print(f"Loaded {len(all_inputs)} samples")
    print(f"Input shape: {all_inputs.shape}")
    print(f"Label shape: {all_labels.shape}")

else:
    # ── Generate ─────────────────────────────────────────────────────────────
    print("Precomputing clean metrics once...")
    clean_metrics_list = []
    for i, clip in enumerate(clean_clips):
        clean_metrics_list.append(extract_metrics_67d(clip))
        print(f"  Clean metrics[{i}]: done")

    print(f"\nGenerating degradations across {len(FREQ_CURRICULUM)} frequencies...")

    all_inputs = []
    all_labels = []
    all_clip_ids = []

    for i, clip in enumerate(clean_clips):
        clean_m = clean_metrics_list[i]

        for f_idx, freq_hz in enumerate(FREQ_CURRICULUM):
            freq_val = freq_to_input(freq_hz)
            for _ in range(DEGRADATIONS_PER_CLIP):
                degraded, gain_db = degrade_with_eq(clip, freq_hz)
                degraded = np.clip(degraded, -1.0, 1.0)
                deg_m = extract_metrics_67d(degraded)

                inp = np.concatenate([deg_m, clean_m, [freq_val]])
                all_inputs.append(inp)
                all_labels.append([gain_db])
                all_clip_ids.append(i)

            if (f_idx + 1) % 10 == 0:
                print(f"  Clip {i+1}/{len(clean_clips)}, freq {f_idx+1}/{len(FREQ_CURRICULUM)}")

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_clip_ids = np.array(all_clip_ids)

    # ── Save to cache ────────────────────────────────────────────────────────
    cache_file = DATA_DIR / 'dataset.npz'
    np.savez_compressed(cache_file, inputs=all_inputs, labels=all_labels, clip_ids=all_clip_ids)
    print(f"\nSaved to: {cache_file}")

print(f"\nTotal samples: {len(all_inputs)}")
print(f"Input shape:  {all_inputs.shape} (should be [N, 135])")
print(f"Label shape:  {all_labels.shape}")
print(f"Label range:  [{all_labels.min():.2f}, {all_labels.max():.2f}] dB")
