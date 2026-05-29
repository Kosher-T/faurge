t_total = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Pass 1: Discover clips and write chosen_pristine.jsonl
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PASS 1: Discovering clips")
print("=" * 60)

clusters = json.load(open(CLUSTER_DATA / 'clusters.json'))
identity_floors = json.load(open(CLUSTER_DATA / 'identity_floors.json'))
print(f"  Loaded {len(clusters)} speakers, {len(set(c['cluster'] for c in clusters.values()))} clusters")

JSONL_PATH = DATASET_DIR / 'chosen_pristine.jsonl'

if JSONL_PATH.exists():
    print(f"  Found existing {JSONL_PATH}")
    chosen = readChosenPristine(JSONL_PATH)
    print(f"  Loaded {len(chosen)} entries from JSONL")
else:
    print("  Scanning datasets...")
    clips = discoverClips(PRISTINE, clusters)
    writeChosenPristine(clips, JSONL_PATH)
    print(f"  Wrote {len(clips)} entries to {JSONL_PATH}")
    # Read back so chosen entries have pair_id
    chosen = readChosenPristine(JSONL_PATH)

# Summary stats
by_dataset = defaultdict(int)
by_speaker = defaultdict(int)
by_cluster = defaultdict(int)
for entry in chosen:
    sid = entry['speaker_id']
    by_speaker[sid] += 1
    by_cluster[entry['cluster_id']] += 1
    if sid.startswith('p'):
        by_dataset['vctk'] += 1
    elif sid.startswith('LJ'):
        by_dataset['ljspeech'] += 1
    else:
        by_dataset['daps'] += 1

print(f"\n  Dataset breakdown:")
for ds, count in sorted(by_dataset.items()):
    print(f"    {ds}: {count} clips")
print(f"  Speakers: {len(by_speaker)}")
print(f"  Cluster distribution:")
for cid in sorted(by_cluster.keys()):
    print(f"    Cluster {cid}: {by_cluster[cid]} clips")

# ══════════════════════════════════════════════════════════════════════════════
# Pass 2: Degrade clips and save
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  PASS 2: Degrading clips")
print("=" * 60)

# Load cluster centroids for metric → cluster assignment
CENTROIDS_PATH = CLUSTER_DATA / 'cluster_centroids.json'
centroids, thresholds = loadClusterCentroids(CENTROIDS_PATH)
print(f"  Loaded {len(centroids)} cluster centroids")

# Check for checkpoint
CHECKPOINT_PATH = DATASET_DIR / 'checkpoint.json'
checkpoint = loadCheckpoint(CHECKPOINT_PATH)
start_idx = checkpoint['next_idx']

# Check what's already done
completed_ids = getCompletedPairIds(DATASET_DIR)
print(f"  Found {len(completed_ids)} already-completed pairs")
print(f"  Resuming from index {start_idx}")

t_start = time.time()
processed = checkpoint.get('processed', len(completed_ids))
failed = 0
all_params = {}
skipped = 0

for idx in range(start_idx, len(chosen)):
    entry = chosen[idx]
    pair_id = entry['pair_id']

    # Skip if already processed
    if pair_id in completed_ids:
        skipped += 1
        continue

    # Load pristine audio
    audio = loadAndPrepareClip(entry['path'])
    if audio is None:
        print(f"  [WARN] {pair_id}: clip too short, skipping")
        failed += 1
        continue

    # Extract 67D metrics → assign cluster_id_pristine
    try:
        metrics_pristine = extract_metrics_67d(audio, SR)
        cluster_id_pristine, is_unknown_pristine = assignCluster(
            metrics_pristine, centroids, thresholds)
    except Exception as e:
        print(f"  [WARN] {pair_id}: metric extraction failed: {e}")
        failed += 1
        continue

    # Generate degradation params
    params = generateDegradationParams()

    # Apply degradation
    try:
        degraded = applyDegradation(audio, SR, params)
    except Exception as e:
        print(f"  [WARN] {pair_id}: degradation failed: {e}")
        failed += 1
        continue

    # Extract 67D metrics of degraded → assign cluster_id_degraded
    try:
        metrics_degraded = extract_metrics_67d(degraded, SR)
        cluster_id_degraded, is_unknown_degraded = assignCluster(
            metrics_degraded, centroids, thresholds)
    except Exception as e:
        print(f"  [WARN] {pair_id}: degraded metric extraction failed: {e}")
        failed += 1
        continue

    # Add cluster IDs to params
    params['cluster_id_pristine'] = cluster_id_pristine
    params['cluster_id_degraded'] = cluster_id_degraded
    params['is_unknown_pristine'] = is_unknown_pristine
    params['is_unknown_degraded'] = is_unknown_degraded
    params['speaker_id'] = entry['speaker_id']
    params['pristine_path'] = entry['path']

    # Save degraded pair
    try:
        saveDegradedPair(pair_id, degraded, params, DATASET_DIR)
        all_params[pair_id] = params
        processed += 1
    except Exception as e:
        print(f"  [WARN] {pair_id}: save failed: {e}")
        failed += 1
        continue

    # Progress report
    if processed % 100 == 0:
        elapsed = time.time() - t_start
        rate = (processed - len(completed_ids)) / elapsed if elapsed > 0 else 0
        print(f"  [{processed}/{len(chosen)}] "
              f"{rate:.1f} pairs/sec, "
              f"{getOutputSizeGB(DATASET_DIR):.1f} GB used, "
              f"failed: {failed}")

    # Checkpoint
    if processed % CHECKPOINT_INTERVAL == 0:
        saveCheckpoint(CHECKPOINT_PATH, {
            'next_idx': idx + 1,
            'processed': processed,
            'failed': failed,
        })

    # Budget check
    if getOutputSizeGB(DATASET_DIR) > MAX_OUTPUT_GB:
        print(f"\n  Budget limit reached ({MAX_OUTPUT_GB} GB). Stopping.")
        break

    # Free memory
    del audio, degraded, metrics_pristine, metrics_degraded
    gc.collect()

# Final checkpoint
saveCheckpoint(CHECKPOINT_PATH, {
    'next_idx': len(chosen),
    'processed': processed,
    'failed': failed,
})

# ══════════════════════════════════════════════════════════════════════════════
# Save metadata
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Saving metadata")
print("=" * 60)

clip_stats = {
    'total_discovered': len(chosen),
    'by_dataset': dict(by_dataset),
    'by_cluster': {str(k): v for k, v in by_cluster.items()},
    'n_speakers': len(by_speaker),
}

saveMetadata(clusters, identity_floors, processed, clip_stats, DATASET_DIR)

# Save degradation params
with open(DATASET_DIR / 'degradation_params.json', 'w') as f:
    json.dump(makeJsonSafe(all_params), f, indent=2)

elapsed = time.time() - t_total
print(f"\n{'=' * 60}")
print(f"  PHASE 2 COMPLETE — {elapsed / 60:.1f} min")
print(f"  Discovered: {len(chosen)} clips")
print(f"  Processed: {processed} (skipped: {skipped}, failed: {failed})")
print(f"  Output: {DATASET_DIR}")
print(f"  Size: {getOutputSizeGB(DATASET_DIR):.2f} GB")
print(f"{'=' * 60}")
