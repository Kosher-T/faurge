t_total = time.time()

# ── Step 1: Load Phase 1 outputs ──
print("═══ Step 1: Loading cluster data ═══")
clusters = json.load(open(CLUSTER_DATA / 'clusters.json'))
identity_floors = json.load(open(CLUSTER_DATA / 'identity_floors.json'))
print(f"  Loaded {len(clusters)} speakers, {len(set(c['cluster'] for c in clusters.values()))} clusters")

# ── Step 2: Load and segment all pristine audio (with cache) ──
print("\n═══ Step 2: Loading pristine audio ═══")
CLIPS_CACHE = DATASET_DIR / 'all_clips_cache.npz'
if CLIPS_CACHE.exists():
    print(f"Found cached clips at {CLIPS_CACHE}")
    all_clips = loadAllClipsCache(CLIPS_CACHE, clusters)
else:
    all_clips = loadAllClips(PRISTINE, clusters)
    saveAllClipsCache(all_clips, CLIPS_CACHE)
total_clips = sum(len(v) for v in all_clips.values())
print(f"  {total_clips} clips from {len(all_clips)} speakers")

# ── Step 3: Generate pair list ──
print("\n═══ Step 3: Generating pairs ═══")
pairs, src_counts, ref_counts = generatePairs(all_clips, clusters, N_PAIRS)
print(f"  {len(pairs)} pairs generated")
print(f"  Source balance: min={min(src_counts.values())}, max={max(src_counts.values())}")
print(f"  Ref balance: min={min(ref_counts.values())}, max={max(ref_counts.values())}")

# ── Step 4: Process pairs in batches ──
print("\n═══ Step 4: Processing degradation pipeline ═══")
t_start = time.time()
processed = 0
skipped = 0

for batch_start in range(0, len(pairs), PAIRS_PER_BATCH):
    batch = pairs[batch_start:batch_start + PAIRS_PER_BATCH]

    for pair in batch:
        # Load source and reference clips from memory
        src_clip = all_clips[pair['source_speaker']][pair['src_clip_idx']]
        ref_clip = all_clips[pair['ref_speaker']][pair['ref_clip_idx']]

        # Generate random degradation parameters
        params = generateDegradationParams()

        # Apply degradation pipeline to source
        try:
            degraded = applyDegradation(src_clip, SR, params)
        except Exception as e:
            print(f"  [WARN] Pair {pair['pair_id']} failed: {e}")
            skipped += 1
            continue

        # Save pair
        savePair(pair['pair_id'], degraded, ref_clip, params, DATASET_DIR)
        processed += 1

    # Progress report
    elapsed = time.time() - t_start
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"  Batch {batch_start // PAIRS_PER_BATCH + 1}: "
          f"{processed}/{len(pairs)} pairs, "
          f"{rate:.1f} pairs/sec, "
          f"{getOutputSizeGB():.1f} GB used")

    # Budget check
    if getOutputSizeGB() > MAX_OUTPUT_GB:
        print(f"\n⚠ Output size limit reached ({MAX_OUTPUT_GB} GB). Stopping.")
        break

    gc.collect()

# ── Step 5: Save metadata ──
print("\n═══ Step 5: Saving metadata ═══")
saveMetadata(clusters, identity_floors, pairs[:processed], src_counts, ref_counts, DATASET_DIR)

elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  PHASE 2 COMPLETE — {elapsed/60:.1f} min")
print(f"  Pairs: {processed} (skipped: {skipped})")
print(f"  Output: {DATASET_DIR}")
print(f"  Size: {getOutputSizeGB():.2f} GB")
print(f"{'='*60}")