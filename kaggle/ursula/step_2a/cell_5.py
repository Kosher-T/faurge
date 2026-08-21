# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — Generate Dataset (with per-dim normalization)
# ══════════════════════════════════════════════════════════════════════════════
# Input: degraded metrics (81D, normalized)
# Output: clean metrics (81D, normalized)
# Normalizer: per-dim min-max from clean metrics, saved for denormalization.

import time
import json

t_start = time.time()

print("Computing clean metrics...")
clean_metrics_list = []
for i, clip in enumerate(clean_clips):
    clean_metrics_list.append(extract_metrics_81d(clip))
    print(f"  Clip {i+1}/{len(clean_clips)}: done")

print(f"\nGenerating {DEGRADATIONS_PER_CLIP} EQ degradations per clip...")

all_degraded = []
all_clean = []

for i, clip in enumerate(clean_clips):
    clean_m = clean_metrics_list[i]

    for _ in range(DEGRADATIONS_PER_CLIP):
        degraded = degrade_with_eq(clip)
        deg_m = extract_metrics_81d(degraded)

        all_degraded.append(deg_m)
        all_clean.append(clean_m)

    print(f"  Clip {i+1}/{len(clean_clips)}: {DEGRADATIONS_PER_CLIP} samples")

all_degraded = np.array(all_degraded, dtype=np.float32)
all_clean = np.array(all_clean, dtype=np.float32)

# ── Compute normalizer from clean metrics ─────────────────────────────────────

metric_min = all_clean.min(axis=0)
metric_max = all_clean.max(axis=0)
metric_range = metric_max - metric_min + 1e-8

normalizer = {
    'min': metric_min.tolist(),
    'max': metric_max.tolist(),
    'range': metric_range.tolist(),
}
with open(DATA_DIR / 'normalizer.json', 'w') as f:
    json.dump(normalizer, f)

# ── Normalize ─────────────────────────────────────────────────────────────────

all_degraded_norm = (all_degraded - metric_min) / metric_range
all_clean_norm = (all_clean - metric_min) / metric_range

elapsed = time.time() - t_start
print(f"\nGeneration complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Degraded shape: {all_degraded_norm.shape}")
print(f"Clean shape:    {all_clean_norm.shape}")
print(f"Normalizer saved to: {DATA_DIR / 'normalizer.json'}")

# ── Verify ────────────────────────────────────────────────────────────────────
print(f"\nAfter normalization:")
print(f"  Degraded range: [{all_degraded_norm.min():.3f}, {all_degraded_norm.max():.3f}]")
print(f"  Clean range:    [{all_clean_norm.min():.3f}, {all_clean_norm.max():.3f}]")
print(f"  Clean mean per dim: [{all_clean_norm.mean(axis=0)[:5].round(3)} ... {all_clean_norm.mean(axis=0)[-5:].round(3)}]")
