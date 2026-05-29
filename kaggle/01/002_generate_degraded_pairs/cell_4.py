t_total = time.time()

# ── Step 1: Load Phase 1 outputs ──
print("═══ Step 1: Loading cluster data ═══")
clusters = json.load(open(CLUSTER_DATA / 'clusters.json'))
identity_floors = json.load(open(CLUSTER_DATA / 'identity_floors.json'))
print(f"  Loaded {len(clusters)} speakers, {len(set(c['cluster'] for c in clusters.values()))} clusters")

# ── Step 2: Load pristine audio (stream to disk) ──
print("\n═══ Step 2: Loading pristine audio ═══")
CLIPS_CACHE = DATASET_DIR / 'clips_cache'
if (CLIPS_CACHE / 'clip_index.json').exists():
    print(f"Found cached clips at {CLIPS_CACHE}")
else:
    print("Building clip cache (streaming to disk)...")
    buildClipCache(PRISTINE, clusters, CLIPS_CACHE)
clip_store = ClipStore(CLIPS_CACHE, clusters)
total_clips = sum(len(clip_store[sid]) for sid in clip_store.keys())
print(f"  {total_clips} clips from {len(list(clip_store.keys()))} speakers")

# ── Step 3: Generate or load pair list (with cache) ──
print("\n═══ Step 3: Generating pairs ═══")
PAIRS_CACHE = DATASET_DIR / 'pairs.json'
if PAIRS_CACHE.exists():
    print(f"Found cached pair list at {PAIRS_CACHE}")
    pairs, src_counts, ref_counts = loadPairList(PAIRS_CACHE)
    print(f"  Loaded {len(pairs)} pairs from cache")
else:
    pairs, src_counts, ref_counts = generatePairs(clip_store, clusters, N_PAIRS)
    savePairList(pairs, src_counts, ref_counts, PAIRS_CACHE)
    print(f"  Generated {len(pairs)} pairs")
print(f"  Source balance: min={min(src_counts.values())}, max={max(src_counts.values())}")
print(f"  Ref balance: min={min(ref_counts.values())}, max={max(ref_counts.values())}")

# ── Step 4: Process pairs in batches (with resume) ──
print("\n═══ Step 4: Processing degradation pipeline ═══")
completed_pairs = getCompletedPairs(DATASET_DIR)
print(f"  Found {len(completed_pairs)} already-completed pairs")

t_start = time.time()
processed = len(completed_pairs)
skipped = 0

for batch_start in range(0, len(pairs), PAIRS_PER_BATCH):
    batch = pairs[batch_start:batch_start + PAIRS_PER_BATCH]

    for pair in batch:
        pair_id = pair['pair_id']

        # Skip if already processed
        if pair_id in completed_pairs:
            continue

        # Load source and reference clips on-demand
        src_clip = clip_store[pair['source_speaker']][pair['src_clip_idx']]
        ref_clip = clip_store[pair['ref_speaker']][pair['ref_clip_idx']]

        # Generate random degradation parameters
        params = generateDegradationParams()

        # Apply degradation pipeline to source
        try:
            degraded = applyDegradation(src_clip, SR, params)
        except Exception as e:
            print(f"  [WARN] Pair {pair_id} failed: {e}")
            skipped += 1
            continue

        # Save pair
        savePair(pair_id, degraded, ref_clip, params, DATASET_DIR)
        processed += 1

        # Clear clip cache periodically to free memory
        if processed % 100 == 0:
            clip_store.clear_cache()
            gc.collect()

    # Progress report
    elapsed = time.time() - t_start
    rate = (processed - len(completed_pairs)) / elapsed if elapsed > 0 else 0
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