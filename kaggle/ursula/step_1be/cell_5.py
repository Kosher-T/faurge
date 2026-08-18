# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Generate Datasets
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 dataset (metric literacy): EQ degradation
# Phase 2 dataset (EQ prediction): 1 EQ band + global gain
# Eval datasets: 2×2 matrix (seen/unseen clips × seen/unseen freqs)
#
# Caching: set PHASE1_CACHE / PHASE2_CACHE in cell_1 to load from Kaggle
# input datasets. Eval datasets are resumable via partial saves — if a run
# times out, re-upload the working dir as a Kaggle dataset and the script
# will pick up where it left off.

import os
import time
import json

# ── Compute normalizer from reference 81D metrics ───────────────────────────
print("Computing reference metrics for all clips...")
ref_metrics_raw = []
for i, clip in enumerate(all_clips):
    ref_metrics_raw.append(compute_metrics_81d(clip))
    print(f"  Clip {i+1}/{len(all_clips)}: done")

ref_metrics_raw = np.array(ref_metrics_raw, dtype=np.float32)

# Per-dim min-max from reference metrics
metric_min = ref_metrics_raw.min(axis=0)
metric_max = ref_metrics_raw.max(axis=0)
metric_range = metric_max - metric_min + 1e-8

# Save normalizer
normalizer = {
    'min': metric_min.tolist(),
    'max': metric_max.tolist(),
    'range': metric_range.tolist(),
}
with open(NORMALIZER_PATH, 'w') as f:
    json.dump(normalizer, f)
print(f"Normalizer saved to: {NORMALIZER_PATH}")

def normalize_metrics(m):
    return (m - metric_min) / metric_range

# Precompute normalized reference metrics for all clips
ref_metrics_norm = [normalize_metrics(m) for m in ref_metrics_raw]

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Metric Literacy Dataset (EQ degradation)
# ══════════════════════════════════════════════════════════════════════════════
# EQ degradation changes Tier 1 features (spectral shape), unlike gain-only.
# Model learns to reconstruct clean metrics from degraded metrics.

PHASE1_FILE = DATA_DIR / 'phase1_dataset.npz'

if PHASE1_CACHE and os.path.exists(PHASE1_CACHE):
    print(f"\nPhase 1: loading from cache: {PHASE1_CACHE}")
    data = np.load(PHASE1_CACHE)
    p1_inputs = data['inputs']
    p1_targets = data['targets']
    print(f"Loaded {len(p1_inputs):,} samples")
elif os.path.exists(PHASE1_FILE):
    print(f"\nPhase 1: loading existing: {PHASE1_FILE}")
    data = np.load(PHASE1_FILE)
    p1_inputs = data['inputs']
    p1_targets = data['targets']
    print(f"Loaded {len(p1_inputs):,} samples")
else:
    t_start = time.time()
    p1_inputs = []
    p1_targets = []

    print(f"\nPhase 1: {len(train_clips)} clips × {DEGRADATIONS_PER_CLIP} EQ degradations")
    for i, clip in enumerate(train_clips):
        ref_norm = ref_metrics_norm[i]
        for _ in range(DEGRADATIONS_PER_CLIP):
            degraded, _, _ = degrade_eq_only(clip, TRAIN_FREQS)
            deg_norm = normalize_metrics(compute_metrics_81d(degraded))

            inp = np.concatenate([deg_norm, ref_norm])  # 162D
            p1_inputs.append(inp)
            p1_targets.append(ref_norm)  # 81D

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(train_clips) - i - 1) / rate if rate > 0 else 0
        print(f"  Clip {i+1}/{len(train_clips)}: done | {elapsed:.0f}s | ~{remaining:.0f}s remaining")

    p1_inputs = np.array(p1_inputs, dtype=np.float32)
    p1_targets = np.array(p1_targets, dtype=np.float32)

    np.savez_compressed(PHASE1_FILE, inputs=p1_inputs, targets=p1_targets)
    elapsed = time.time() - t_start
    print(f"Phase 1 dataset: {p1_inputs.shape} | saved in {elapsed:.0f}s")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: EQ Prediction Dataset (1 EQ band + global gain)
# ══════════════════════════════════════════════════════════════════════════════
# Resumable: saves partial progress every 50 combos. To resume across
# sessions, upload the working dir as a Kaggle dataset and set
# PHASE2_PARTIAL_LOAD in cell_1.

PHASE2_FILE = DATA_DIR / 'phase2_dataset.npz'
PHASE2_PARTIAL = DATA_DIR / 'phase2_partial.npz'
SAVE_EVERY = 50

if PHASE2_CACHE and os.path.exists(PHASE2_CACHE):
    print(f"\nPhase 2: loading from cache: {PHASE2_CACHE}")
    data = np.load(PHASE2_CACHE)
    p2_inputs = data['inputs']
    p2_targets = data['targets']
    print(f"Loaded {len(p2_inputs):,} samples")
elif os.path.exists(PHASE2_FILE):
    print(f"\nPhase 2: loading existing: {PHASE2_FILE}")
    data = np.load(PHASE2_FILE)
    p2_inputs = data['inputs']
    p2_targets = data['targets']
    print(f"Loaded {len(p2_inputs):,} samples")
else:
    # Try loading from partial (Kaggle input or working dir)
    done_combos = set()
    all_freq_hz = []
    all_clip_ids = []
    p2_inputs = []
    p2_targets = []
    partial_source = None

    if PHASE2_PARTIAL_LOAD and os.path.exists(PHASE2_PARTIAL_LOAD):
        partial_source = PHASE2_PARTIAL_LOAD
    elif os.path.exists(PHASE2_PARTIAL):
        partial_source = PHASE2_PARTIAL

    if partial_source:
        print(f"Phase 2: loading from: {partial_source}")
        data = np.load(partial_source)
        p2_inputs = list(data['inputs'])
        p2_targets = list(data['targets'])
        if 'freq_hz' in data and 'clip_ids' in data:
            all_freq_hz = list(data['freq_hz'])
            all_clip_ids = list(data['clip_ids'])
            for cid in np.unique(data['clip_ids']):
                for fid in np.unique(data['freq_hz'][data['clip_ids'] == cid]):
                    done_combos.add((int(cid), float(fid)))
            print(f"  Resuming: {len(done_combos)} combos done, {len(p2_inputs):,} samples")
        else:
            # Complete dataset without tracking keys — save as final and skip generation
            p2_inputs = np.array(p2_inputs, dtype=np.float32)
            p2_targets = np.array(p2_targets, dtype=np.float32)
            np.savez_compressed(PHASE2_FILE, inputs=p2_inputs, targets=p2_targets)
            print(f"  Complete dataset: {len(p2_inputs):,} samples → saved to {PHASE2_FILE}")
            done_combos = None  # signal: data is complete, skip generation

    # Generate remaining combos (if any)
    total_combos = len(train_clips) * len(TRAIN_FREQS)
    if done_combos is not None and len(done_combos) < total_combos:
        t_start = time.time()
        print(f"\nPhase 2: {len(train_clips)} clips × {len(TRAIN_FREQS)} freqs × {DEGRADATIONS_PER_CLIP}")
        combo_count = len(done_combos)
        save_count = 0

        for i, clip in enumerate(train_clips):
            ref_norm = ref_metrics_norm[i]
            for freq_hz in TRAIN_FREQS:
                if (i, float(freq_hz)) in done_combos:
                    continue

                for _ in range(DEGRADATIONS_PER_CLIP):
                    degraded, targets = degrade_eq_gain(clip, TRAIN_FREQS)
                    deg_norm = normalize_metrics(compute_metrics_81d(degraded))

                    inp = np.concatenate([deg_norm, ref_norm])  # 162D
                    p2_inputs.append(inp)
                    p2_targets.append(targets)  # 3D
                    all_freq_hz.append(freq_hz)
                    all_clip_ids.append(i)

                combo_count += 1
                save_count += 1

                if save_count % SAVE_EVERY == 0:
                    tmp_inputs = np.array(p2_inputs, dtype=np.float32)
                    tmp_targets = np.array(p2_targets, dtype=np.float32)
                    tmp_freqs = np.array(all_freq_hz, dtype=np.float32)
                    tmp_clips = np.array(all_clip_ids, dtype=np.int32)
                    np.savez_compressed(PHASE2_PARTIAL,
                        inputs=tmp_inputs, targets=tmp_targets,
                        freq_hz=tmp_freqs, clip_ids=tmp_clips)

                    elapsed = time.time() - t_start
                    rate = save_count / elapsed if elapsed > 0 else 0
                    remaining = (total_combos - combo_count) / rate if rate > 0 else 0
                    print(f"  [{combo_count}/{total_combos}] {len(p2_inputs):,} samples | "
                          f"{elapsed:.0f}s | ~{remaining:.0f}s remaining")

        p2_inputs = np.array(p2_inputs, dtype=np.float32)
        p2_targets = np.array(p2_targets, dtype=np.float32)

        np.savez_compressed(PHASE2_FILE, inputs=p2_inputs, targets=p2_targets)
        elapsed = time.time() - t_start
        print(f"Phase 2 dataset: {p2_inputs.shape} | saved in {elapsed:.0f}s")

        if os.path.exists(PHASE2_PARTIAL):
            os.remove(PHASE2_PARTIAL)
    elif done_combos is not None and isinstance(p2_inputs, list):
        # All combos were already done (loaded from partial), convert to arrays
        p2_inputs = np.array(p2_inputs, dtype=np.float32)
        p2_targets = np.array(p2_targets, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Eval Datasets: 2×2 Matrix (resumable)
# ══════════════════════════════════════════════════════════════════════════════

def gen_eval_dataset(clips, clip_indices, freq_pool, n_per_combo, out_file, partial_file, label):
    """Generate eval dataset. Resumes from partial if it exists."""
    if os.path.exists(out_file):
        print(f"  [{label}] loading cached: {out_file}")
        data = np.load(out_file)
        return data['inputs'], data['labels']

    # Check for resumable partial
    done_combos = set()
    inputs = []
    labels = []
    partial_freqs = []
    partial_clips = []
    if os.path.exists(partial_file):
        print(f"  [{label}] resuming from partial: {partial_file}")
        data = np.load(partial_file)
        inputs = list(data['inputs'])
        labels = list(data['labels'])
        partial_freqs = list(data['freq_hz'])
        partial_clips = list(data['clip_ids'])
        for cid in np.unique(data['clip_ids']):
            for fid in np.unique(data['freq_hz'][data['clip_ids'] == cid]):
                done_combos.add((int(cid), float(fid)))
        print(f"    Already done: {len(done_combos)} combos, {len(inputs):,} samples")

    print(f"  [{label}] generating {len(clips)} clips × {len(freq_pool)} freqs × {n_per_combo}...")
    t0 = time.time()
    combo_count = len(done_combos)
    save_count = 0

    for i, clip in enumerate(clips):
        ref_norm = ref_metrics_norm[clip_indices[i]]
        for freq_hz in freq_pool:
            if (i, float(freq_hz)) in done_combos:
                continue
            for _ in range(n_per_combo):
                degraded, targets = degrade_eq_gain(clip, freq_pool)
                deg_norm = normalize_metrics(compute_metrics_81d(degraded))
                inputs.append(np.concatenate([deg_norm, ref_norm]))
                labels.append(targets)
                partial_freqs.append(freq_hz)
                partial_clips.append(i)

            combo_count += 1
            save_count += 1

            if save_count % 20 == 0:
                tmp_inputs = np.array(inputs, dtype=np.float32)
                tmp_labels = np.array(labels, dtype=np.float32)
                tmp_freqs = np.array(partial_freqs, dtype=np.float32)
                tmp_clips = np.array(partial_clips, dtype=np.int32)
                np.savez_compressed(partial_file,
                    inputs=tmp_inputs, labels=tmp_labels,
                    freq_hz=tmp_freqs, clip_ids=tmp_clips)
                elapsed = time.time() - t0
                print(f"    [{combo_count}/{len(clips)*len(freq_pool)}] saved | {elapsed:.0f}s")

    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    np.savez_compressed(out_file, inputs=inputs, labels=labels)
    elapsed = time.time() - t0
    print(f"    saved {len(inputs):,} samples ({elapsed:.0f}s)")

    if os.path.exists(partial_file):
        os.remove(partial_file)

    return inputs, labels

print(f"\nGenerating eval datasets (2×2 matrix)...")

eval_seen_seen = gen_eval_dataset(
    train_clips, list(range(len(train_clips))), TRAIN_FREQS, EVAL_DEGRADATIONS_PER_CLIP,
    DATA_DIR / 'eval_seen_seen.npz', DATA_DIR / 'eval_seen_seen_partial.npz',
    'seen clip × seen freq')

eval_seen_unseen = gen_eval_dataset(
    train_clips, list(range(len(train_clips))), UNSEEN_FREQS, EVAL_DEGRADATIONS_PER_CLIP,
    DATA_DIR / 'eval_seen_unseen.npz', DATA_DIR / 'eval_seen_unseen_partial.npz',
    'seen clip × unseen freq')

test_offset = len(train_clips)
eval_unseen_seen = gen_eval_dataset(
    test_clips, list(range(test_offset, test_offset + len(test_clips))),
    TRAIN_FREQS, EVAL_DEGRADATIONS_PER_CLIP,
    DATA_DIR / 'eval_unseen_seen.npz', DATA_DIR / 'eval_unseen_seen_partial.npz',
    'unseen clip × seen freq')

eval_unseen_unseen = gen_eval_dataset(
    test_clips, list(range(test_offset, test_offset + len(test_clips))),
    UNSEEN_FREQS, EVAL_DEGRADATIONS_PER_CLIP,
    DATA_DIR / 'eval_unseen_unseen.npz', DATA_DIR / 'eval_unseen_unseen_partial.npz',
    'unseen clip × unseen freq')

print(f"\nAll datasets generated.")
