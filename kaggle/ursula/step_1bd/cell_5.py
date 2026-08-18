# ══════════════════════════════════════════════════════════════════════════════
# Step 1bd — Generate or Load Dataset (RESUMABLE, NORMALIZED)
# ══════════════════════════════════════════════════════════════════════════════
# Input: 160D = [degraded_80D_norm, clean_80D_norm]
# Uses step_1bc normalizer (data-dependent min-max) for per-dim scaling.

import os
import time
import json

# ── Load normalizer from A_step_1bc ───────────────────────────────────────────

with open(STEP1BC_NORMALIZER) as f:
    normalizer = json.load(f)
metric_min = np.array(normalizer['min'], dtype=np.float32)
metric_max = np.array(normalizer['max'], dtype=np.float32)
metric_range = np.array(normalizer['range'], dtype=np.float32)
print(f"Loaded normalizer from A_step_1bc: {len(metric_min)}D")

def normalize_metrics(m):
    """Normalize 80D metrics to [0,1] using A_step_1bc stats."""
    return (m - metric_min) / metric_range

PARTIAL_FILE = DATA_DIR / 'dataset_partial.npz'
FINAL_FILE = DATA_DIR / 'dataset.npz'
SAVE_EVERY = 50

if CACHE_PATH and os.path.exists(CACHE_PATH):
    print(f"Loading cached data from: {CACHE_PATH}")
    t0 = time.time()
    data = np.load(CACHE_PATH)
    all_inputs = data['inputs']
    all_labels = data['labels']
    all_freq_hz = data['freq_hz']
    all_clip_ids = data['clip_ids']
    print(f"Loaded {len(all_inputs):,} samples in {time.time()-t0:.1f}s")

elif os.path.exists(FINAL_FILE):
    print(f"Dataset already exists: {FINAL_FILE}")
    data = np.load(FINAL_FILE)
    all_inputs = data['inputs']
    all_labels = data['labels']
    all_freq_hz = data['freq_hz']
    all_clip_ids = data['clip_ids']
    print(f"Loaded {len(all_inputs):,} samples")

else:
    t_start = time.time()

    done_combos = set()
    if os.path.exists(PARTIAL_FILE):
        print(f"Resuming from partial file: {PARTIAL_FILE}")
        data = np.load(PARTIAL_FILE)
        all_inputs = list(data['inputs'])
        all_labels = list(data['labels'])
        all_freq_hz = list(data['freq_hz'])
        all_clip_ids = list(data['clip_ids'])
        for cid in np.unique(data['clip_ids']):
            for fid in np.unique(data['freq_hz'][data['clip_ids'] == cid]):
                done_combos.add((int(cid), float(fid)))
        print(f"  Already done: {len(done_combos)} combos, {len(all_inputs):,} samples")
    else:
        all_inputs = []
        all_labels = []
        all_freq_hz = []
        all_clip_ids = []

    print("Precomputing clean metrics...")
    clean_metrics_list = []
    for i, clip in enumerate(clean_clips):
        raw = extract_metrics_80d(clip)
        clean_metrics_list.append(normalize_metrics(raw))
        print(f"  Clip {i+1}/{len(clean_clips)}: done")

    total_combos = len(clean_clips) * len(FREQ_CURRICULUM)
    print(f"\nGenerating: {len(clean_clips)} clips × {len(FREQ_CURRICULUM)} freqs × {DEGRADATIONS_PER_CLIP} each")

    save_count = 0
    total_generated = 0

    for i, clip in enumerate(clean_clips):
        clean_m = clean_metrics_list[i]

        for f_idx, freq_hz in enumerate(FREQ_CURRICULUM):
            combo_key = (i, float(freq_hz))
            if combo_key in done_combos:
                continue

            for _ in range(DEGRADATIONS_PER_CLIP):
                degraded, gain_db = degrade_with_eq(clip, freq_hz)
                degraded = np.clip(degraded, -1.0, 1.0)
                deg_m = normalize_metrics(extract_metrics_80d(degraded))

                inp = np.concatenate([deg_m, clean_m])
                all_inputs.append(inp)
                all_labels.append([normalize_freq(freq_hz), normalize_gain(gain_db)])
                all_freq_hz.append(freq_hz)
                all_clip_ids.append(i)
                total_generated += 1

            save_count += 1

            if save_count % SAVE_EVERY == 0:
                tmp_inputs = np.array(all_inputs, dtype=np.float32)
                tmp_labels = np.array(all_labels, dtype=np.float32)
                tmp_freqs = np.array(all_freq_hz, dtype=np.float32)
                tmp_clips = np.array(all_clip_ids, dtype=np.int32)
                np.savez_compressed(PARTIAL_FILE,
                    inputs=tmp_inputs, labels=tmp_labels,
                    freq_hz=tmp_freqs, clip_ids=tmp_clips)

                elapsed = time.time() - t_start
                total_done = len(done_combos) + save_count
                rate = save_count / elapsed if elapsed > 0 else 0
                remaining = total_combos - total_done
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{total_done}/{total_combos}] saved | {total_generated:,} samples | "
                      f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_freq_hz = np.array(all_freq_hz, dtype=np.float32)
    all_clip_ids = np.array(all_clip_ids, dtype=np.int32)

    elapsed = time.time() - t_start
    print(f"\nGeneration complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    np.savez_compressed(FINAL_FILE,
        inputs=all_inputs, labels=all_labels,
        freq_hz=all_freq_hz, clip_ids=all_clip_ids)
    print(f"Saved to: {FINAL_FILE}")
    print(f"File size: {os.path.getsize(FINAL_FILE) / 1e6:.1f} MB")

    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)

print(f"\nDataset summary:")
print(f"  Total samples: {len(all_inputs):,}")
print(f"  Input shape:   {all_inputs.shape} (should be [N, {METRIC_DIM * 2}])")
print(f"  Label shape:   {all_labels.shape} (should be [N, 2])")
print(f"  Input range:   [{all_inputs.min():.3f}, {all_inputs.max():.3f}]")
