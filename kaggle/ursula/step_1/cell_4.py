# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Generate Degradations
# ══════════════════════════════════════════════════════════════════════════════
# Apply real plugins to degrade audio. Returns (degraded_metrics, labels).
# Labels = the plugin parameters that were applied.

import sys

# Add plugin directories to sys.path (each plugin is a single .py file in its own dir)
if USE_GAIN:
    sys.path.insert(0, str(PLUGIN_DIR / 'gain1'))
    import gain1 as gain_plugin
if USE_EQ:
    sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
    import equalizer as eq_plugin
if USE_COMPRESSOR:
    sys.path.insert(0, str(PLUGIN_DIR / 'compressor'))
    import compressor as comp_plugin

# ── Degradation functions ─────────────────────────────────────────────────────

def degrade_with_gain(audio, sr=SR):
    """Apply random gain, return (degraded_audio, label_dict)."""
    gain_db = np.random.uniform(*GAIN_RANGE_DB)
    degraded, _ = gain_plugin.process(audio, sr=sr, gain_db=float(gain_db))
    label = {"gain_db": gain_db}
    return degraded, label

def degrade_with_eq(audio, sr=SR):
    """Apply random EQ (N_EQ_BANDS bands), return (degraded_audio, label_dict)."""
    labels = {"eq_bands": []}
    degraded = audio.copy()

    for b in range(N_EQ_BANDS):
        freq = np.random.uniform(*EQ_FREQ_RANGE_HZ)
        gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
        q = np.random.uniform(*EQ_Q_RANGE)

        degraded, _ = eq_plugin.process(
            degraded, sr=sr,
            bands=[{
                "freq_hz": float(freq),
                "gain_db": float(gain_db),
                "q": float(q),
                "filter_type": "peak"
            }]
        )
        labels["eq_bands"].append({"freq_hz": freq, "gain_db": gain_db, "q": q})

    return degraded, labels

def degrade_with_compressor(audio, sr=SR):
    """Apply random compression, return (degraded_audio, label_dict)."""
    threshold_db = np.random.uniform(*COMP_THRESHOLD_RANGE_DB)
    ratio = np.random.uniform(*COMP_RATIO_RANGE)
    attack_ms = np.random.uniform(*COMP_ATTACK_RANGE_MS)
    release_ms = np.random.uniform(*COMP_RELEASE_RANGE_MS)

    degraded, _ = comp_plugin.process(
        audio, sr=sr,
        threshold_db=float(threshold_db),
        ratio=float(ratio),
        attack_ms=float(attack_ms),
        release_ms=float(release_ms)
    )

    label = {
        "comp_threshold_db": threshold_db,
        "comp_ratio": ratio,
        "comp_attack_ms": attack_ms,
        "comp_release_ms": release_ms,
    }
    return degraded, label

# ── Combined degradation ─────────────────────────────────────────────────────

def generate_degradation(audio, sr=SR):
    """Apply enabled plugins in order: gain → EQ → compressor."""
    degraded = audio.copy()
    combined_label = {}

    if USE_GAIN:
        degraded, gain_label = degrade_with_gain(degraded, sr)
        combined_label.update(gain_label)

    if USE_EQ:
        degraded, eq_label = degrade_with_eq(degraded, sr)
        combined_label.update(eq_label)

    if USE_COMPRESSOR:
        degraded, comp_label = degrade_with_compressor(degraded, sr)
        combined_label.update(comp_label)

    return degraded, combined_label

def labels_to_vector(label):
    """Convert label dict to flat numpy vector (order matches OUTPUT_DIM)."""
    parts = []
    if USE_GAIN:
        parts.append([label["gain_db"]])
    if USE_EQ:
        for band in label["eq_bands"]:
            # Log-scale for frequency (standard practice, matches production)
            parts.append([np.log(band["freq_hz"]), band["gain_db"], band["q"]])
    if USE_COMPRESSOR:
        parts.append([
            label["comp_threshold_db"],
            label["comp_ratio"],
            label["comp_attack_ms"],
            label["comp_release_ms"],
        ])
    return np.concatenate(parts).astype(np.float32)

# ── Process all clips ─────────────────────────────────────────────────────────

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
        degraded, label = generate_degradation(clip)
        degraded = np.clip(degraded, -1.0, 1.0)
        metrics = extract_metrics_67d(degraded)
        vec = labels_to_vector(label)

        clip_degraded.append(metrics)
        clip_labels.append(vec)
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
print(f"Label range: [{all_labels.min():.2f}, {all_labels.max():.2f}]")
