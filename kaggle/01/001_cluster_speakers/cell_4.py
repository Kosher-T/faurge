t_total = time.time()

# ── Step 1: Discover speakers ──
print("═══ Step 1: Discovering speakers ═══")
speakers = discover_speakers(PATHS)
total_clips = sum(len(v) for v in speakers.values())
print(f"Found {len(speakers)} speakers, {total_clips} total clips")
for ds in ["vctk", "ljspeech", "daps"]:
    ds_speakers = {k: v for k, v in speakers.items()
                   if (ds == "vctk" and k.startswith("p")) or
                      (ds == "ljspeech" and k.startswith("LJ")) or
                      (ds == "daps" and not k.startswith("p") and not k.startswith("LJ"))}
    if ds_speakers:
        ds_clips = sum(len(v) for v in ds_speakers.values())
        print(f"  {ds}: {len(ds_speakers)} speakers, {ds_clips} clips")

# ── Step 2: Compute speaker profiles (with cache) ──
print("\n═══ Step 2: Computing speaker profiles ═══")
if PROFILES_CACHE.exists():
    print(f"Found cached profiles at {PROFILES_CACHE}")
    profiles = load_profiles(PROFILES_CACHE)
else:
    profiles = compute_speaker_profiles(speakers, SAMPLES_PER_SPEAKER)
    save_profiles(profiles, PROFILES_CACHE)
print(f"Using {len(profiles)} speaker profiles (67D each)")

# ── Step 3: Clustering ──
print("\n═══ Step 3: Clustering ═══")
labels, centroids_67d, thresholds, speaker_ids, X, X_scaled = \
    run_clustering(profiles, N_CLUSTERS)

# ── Step 4: t-SNE visualization ──
print("\n═══ Step 4: t-SNE visualization ═══")
perplexity = min(30, len(speaker_ids) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
X_2d = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
palette = sns.color_palette("tab10", N_CLUSTERS)
for c in range(N_CLUSTERS):
    mask = labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[palette[c]], label=f"Cluster {c}",
               s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

for i, sid in enumerate(speaker_ids):
    ax.annotate(sid, (X_2d[i, 0], X_2d[i, 1]),
                fontsize=7, alpha=0.6, ha="center", va="bottom")

ax.set_title("t-SNE of 67D Speaker Profiles (colored by cluster)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

tsne_path = CLUSTER_DATA_DIR / "tsne_clusters.png"
fig.savefig(tsne_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {tsne_path}")

# ── Step 5: Identity floors ──
print("\n═══ Step 5: Identity floors ═══")
floors = compute_identity_floors(profiles, labels, speaker_ids)

# ── Step 6: Validation ──
validate_results(labels, floors, centroids_67d, thresholds)

# ── Step 7: Save outputs ──
print("\n═══ Step 7: Saving outputs ═══")
save_outputs(profiles, labels, centroids_67d, thresholds, floors, speaker_ids, CLUSTER_DATA_DIR)

elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  PHASE 1 COMPLETE — {elapsed/60:.1f} min")
print(f"  Output: {CLUSTER_DATA_DIR}")
print(f"{'='*60}")
