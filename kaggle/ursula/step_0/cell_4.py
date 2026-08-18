# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Generate Degradations
# ══════════════════════════════════════════════════════════════════════════════
# Apply random gain + EQ in FFT domain. No plugins. Just signal processing.
# Uses variables from cell_1: GAIN_RANGE_DB, EQ_N_BANDS_RANGE, EQ_GAIN_RANGE_DB,
#   EQ_FREQ_RANGE_HZ, EQ_Q_RANGE

def apply_gain_db(audio, gain_db):
    return audio * (10.0 ** (gain_db / 20.0))

def apply_random_eq(audio, n_bands, sr=SR):
    """Apply random parametric EQ in FFT domain."""
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    result = fft.copy()

    for _ in range(n_bands):
        center_hz = np.random.uniform(*EQ_FREQ_RANGE_HZ)
        gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
        q = np.random.uniform(*EQ_Q_RANGE)

        bandwidth = center_hz / q
        low = center_hz - bandwidth / 2
        high = center_hz + bandwidth / 2
        eq_curve = np.ones_like(freqs)
        mask = (freqs >= low) & (freqs <= high)
        eq_curve[mask] = 10.0 ** (gain_db / 20.0 * np.exp(
            -0.5 * ((freqs[mask] - center_hz) / (bandwidth / 2)) ** 2
        ))
        result *= eq_curve

    return np.fft.irfft(result, n=len(audio)).astype(np.float32)

# ── Generate degraded versions ───────────────────────────────────────────────

def generate_degradations(clean_clip, n_degradations=DEGRADATIONS_PER_CLIP):
    """Generate N degraded versions of a single clean clip."""
    degraded_clips = []
    degraded_metrics = []

    for _ in range(n_degradations):
        gain_db = np.random.uniform(*GAIN_RANGE_DB)
        n_eq_bands = np.random.randint(*EQ_N_BANDS_RANGE)

        degraded = apply_gain_db(clean_clip, gain_db)
        degraded = apply_random_eq(degraded, n_eq_bands)
        degraded = np.clip(degraded, -1.0, 1.0)

        metrics = extract_metrics_67d(degraded)
        degraded_clips.append(degraded)
        degraded_metrics.append(metrics)

    return degraded_clips, degraded_metrics

# ── Process all clips ─────────────────────────────────────────────────────────

print("Generating degradations...")

all_clean_metrics = []
all_degraded_metrics = []
all_clip_ids = []

for i, clip in enumerate(clean_clips):
    clean_metrics = extract_metrics_67d(clip)
    all_clean_metrics.append(clean_metrics)

    degraded_clips, degraded_metrics = generate_degradations(clip)
    all_degraded_metrics.extend(degraded_metrics)
    all_clip_ids.extend([i] * len(degraded_metrics))

    print(f"  Clip {i+1}/{len(clean_clips)}: {len(degraded_metrics)} degraded versions")

all_clean_metrics = np.array(all_clean_metrics)
all_degraded_metrics = np.array(all_degraded_metrics)
all_clip_ids = np.array(all_clip_ids)

print(f"\nTotal degraded samples: {len(all_degraded_metrics)}")
print(f"Clean metrics shape: {all_clean_metrics.shape}")
print(f"Degraded metrics shape: {all_degraded_metrics.shape}")
