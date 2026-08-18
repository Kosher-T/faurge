# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Generate Dataset
# ══════════════════════════════════════════════════════════════════════════════

print("Generating degradations...")

all_clean_metrics = []
all_degraded_metrics = []
all_labels = []
all_clip_ids = []

for i, clip in enumerate(clean_clips):
    clean_metrics = extract_metrics_67d(clip)
    all_clean_metrics.append(clean_metrics)

    clip_degraded = []
    clip_labels = []
    clip_ids = []

    for _ in range(DEGRADATIONS_PER_CLIP):
        degraded, gain_db = degrade_with_eq(clip)
        degraded = np.clip(degraded, -1.0, 1.0)
        metrics = extract_metrics_67d(degraded)

        clip_degraded.append(metrics)
        clip_labels.append([gain_db])  # 1D label
        clip_ids.append(i)

    all_degraded_metrics.extend(clip_degraded)
    all_labels.extend(clip_labels)
    all_clip_ids.extend(clip_ids)

    print(f"  Clip {i+1}/{len(clean_clips)}: {len(clip_labels)} degraded versions")

all_clean_metrics = np.array(all_clean_metrics)
all_degraded_metrics = np.array(all_degraded_metrics)
all_labels = np.array(all_labels)
all_clip_ids = np.array(all_clip_ids)

print(f"\nTotal degraded samples: {len(all_degraded_metrics)}")
print(f"Clean metrics shape: {all_clean_metrics.shape}")
print(f"Degraded metrics shape: {all_degraded_metrics.shape}")
print(f"Labels shape: {all_labels.shape}")
print(f"Label range: [{all_labels.min():.2f}, {all_labels.max():.2f}] dB")
