t_total = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Load inputs
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Loading inputs")
print("=" * 60)

clusters = json.load(open(CLUSTER_DATA / 'clusters.json'))
identity_floors = json.load(open(CLUSTER_DATA / 'identity_floors.json'))
centroids = load_centroids(CLUSTER_DATA)
print(f"  Loaded {len(clusters)} speakers, {len(centroids)} centroids")

JSONL_PATH = PAIRS_DATA / 'chosen_pristine.jsonl'
chosen = read_chosen_pristine(JSONL_PATH)
print(f"  Loaded {len(chosen)} pairs from JSONL")

# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint setup
# ══════════════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = METRICS_DIR / 'checkpoint.json'
checkpoint = load_checkpoint(CHECKPOINT_PATH)
completed_ids = get_completed_pair_ids(METRICS_DIR)
start_idx = checkpoint['next_idx']
print(f"  Found {len(completed_ids)} already-completed pairs")
print(f"  Resuming from index {start_idx}")

# ══════════════════════════════════════════════════════════════════════════════
# Extract metrics for each pair
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  Extracting metrics ({len(chosen)} pairs)")
print("=" * 60)

t_start = time.time()
processed = checkpoint.get('processed', len(completed_ids))
failed = 0
skipped = 0
paths_rows = []

for idx in range(start_idx, len(chosen)):
    entry = chosen[idx]
    pair_id = entry['pair_id']

    if pair_id in completed_ids:
        skipped += 1
        continue

    # Load degraded audio
    degraded_path = PAIRS_DATA / 'pairs' / pair_id / f'{pair_id}.wav'
    if not degraded_path.exists():
        print(f"  [WARN] {pair_id}: degraded WAV not found, skipping")
        failed += 1
        continue

    degraded_audio = load_audio(degraded_path)
    if degraded_audio is None:
        print(f"  [WARN] {pair_id}: degraded audio too short, skipping")
        failed += 1
        continue

    # Load pristine audio
    pristine_audio = load_audio(entry['path'])
    if pristine_audio is None:
        print(f"  [WARN] {pair_id}: pristine audio too short, skipping")
        failed += 1
        continue

    # Extract 67D metrics
    try:
        m_degraded = extract_metrics_67d(degraded_audio)
        m_pristine = extract_metrics_67d(pristine_audio)
    except Exception as e:
        print(f"  [WARN] {pair_id}: metric extraction failed: {e}")
        failed += 1
        continue

    # Assign cluster for degraded
    cluster_id_degraded, _ = assign_cluster(m_degraded, centroids)

    # Stack into 2x67 tensor and save
    metrics_tensor = np.stack([m_degraded, m_pristine])  # (2, 67)
    pt_path = PAIRS_OUT / f'{pair_id}.pt'
    torch.save(torch.from_numpy(metrics_tensor), pt_path)

    # Record paths row
    cluster_id_reference = clusters[entry['speaker_id']]['cluster']
    paths_rows.append({
        'pair_id': pair_id,
        'degraded_path': str(degraded_path),
        'reference_path': entry['path'],
        'cluster_id_degraded': cluster_id_degraded,
        'cluster_id_reference': cluster_id_reference,
    })

    processed += 1

    # Progress
    if processed % 10 == 0 or processed == len(chosen):
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  [{processed}/{len(chosen)}] {rate:.1f} pairs/sec, failed: {failed}")

    # Checkpoint
    if processed % 100 == 0:
        save_checkpoint(CHECKPOINT_PATH, {
            'next_idx': idx + 1,
            'processed': processed,
            'failed': failed,
        })

    del degraded_audio, pristine_audio
    gc.collect()

# Final checkpoint
save_checkpoint(CHECKPOINT_PATH, {
    'next_idx': len(chosen),
    'processed': processed,
    'failed': failed,
})

# ══════════════════════════════════════════════════════════════════════════════
# Save paths.csv
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Saving paths.csv")
print("=" * 60)

csv_path = METRICS_DIR / 'paths.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pair_id', 'degraded_path', 'reference_path',
                                              'cluster_id_degraded', 'cluster_id_reference'])
    writer.writeheader()
    writer.writerows(paths_rows)
print(f"  Wrote {len(paths_rows)} rows → {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Save metadata.json
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Saving metadata.json")
print("=" * 60)

cluster_assignments = {sid: info['cluster'] for sid, info in clusters.items()}
metadata = {
    'version': 1,
    'n_pairs': processed,
    'n_failed': failed,
    'sample_rate': SR,
    'clip_seconds': CLIP_SEC,
    'metrics_dim': 67,
    'metrics_order': 'LTAS(64) + LUFS(1) + Crest(1) + ZCR(1)',
    'cluster_floors': identity_floors,
    'cluster_assignments': cluster_assignments,
    'n_clusters': N_CLUSTERS,
}
with open(METRICS_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved {METRICS_DIR / 'metadata.json'}")

elapsed = time.time() - t_total
print(f"\n{'=' * 60}")
print(f"  PHASE 3 COMPLETE — {elapsed / 60:.1f} min")
print(f"  Processed: {processed} (skipped: {skipped}, failed: {failed})")
print(f"  Output: {METRICS_DIR}")
print(f"{'=' * 60}")
