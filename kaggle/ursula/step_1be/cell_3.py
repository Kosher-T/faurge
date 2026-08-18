# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Metric Extraction (81D)
# ══════════════════════════════════════════════════════════════════════════════
# Tier 0 (67D): LTAS64 + LUFS + Crest + ZCR
# Tier 1 (13D): Centroid, Bandwidth, Flatness, Flux, Rolloff, Skewness,
#               Kurtosis, Slope + Band Energy Ratios (5)
# Extra  (1D):  RMS Energy (Tier 1 #17)
# Total: 81D per audio clip

import scipy.signal

# ── Tier 0: Original 67D ────────────────────────────────────────────────────

def compute_ltas(audio, sr=SR, fft_size=FFT_SIZE, hop_size=HOP_SIZE, n_bands=LTAS_DIM):
    freqs, times, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=fft_size, noverlap=fft_size - hop_size)
    power = np.abs(Zxx) ** 2
    low_bin = max(int(BARK_LOW_HZ / (sr / fft_size)), 1)
    high_bin = min(int(BARK_HIGH_HZ / (sr / fft_size)), power.shape[0])
    power = power[low_bin:high_bin, :]
    band_edges = np.linspace(0, power.shape[0], n_bands + 1, dtype=int)
    ltas = np.zeros(n_bands)
    for i in range(n_bands):
        band_power = power[band_edges[i]:band_edges[i + 1], :]
        ltas[i] = np.mean(band_power) if band_power.size > 0 else 1e-10
    ltas = np.log10(ltas + 1e-10)
    return ltas

def compute_lufs(audio, sr=SR):
    return np.clip(-0.691 + 10 * np.log10(np.mean(audio ** 2) + 1e-10), -70, 0)

def compute_crest(audio, frame_size=2048, hop_size=512):
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, hop_size)]
    if not frames:
        return 0.0
    rms = np.sqrt(np.mean(np.array(frames) ** 2, axis=1) + 1e-10)
    peak = np.max(np.abs(np.array(frames)), axis=1) + 1e-10
    return np.mean(20 * np.log10(peak / rms + 1e-10))

def compute_zcr(audio, frame_size=2048, hop_size=512):
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, hop_size)]
    if not frames:
        return 0.0
    return np.mean(np.sum(np.abs(np.diff(np.sign(np.array(frames)))), axis=1) / (2 * frame_size))

# ── Tier 1: Spectral features (computed from STFT) ─────────────────────────

def compute_stft(audio, sr=SR, fft_size=FFT_SIZE, hop_size=HOP_SIZE):
    freqs, times, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=fft_size, noverlap=fft_size - hop_size)
    mag = np.abs(Zxx)
    return freqs, mag

def compute_spectral_centroid(freqs, mag):
    power = mag ** 2
    total_power = np.sum(power, axis=0) + 1e-10
    centroid = np.sum(freqs[:, None] * power, axis=0) / total_power
    return np.mean(centroid)

def compute_spectral_bandwidth(freqs, mag, centroid_hz):
    power = mag ** 2
    total_power = np.sum(power, axis=0) + 1e-10
    bw = np.sqrt(np.sum(((freqs[:, None] - centroid_hz) ** 2) * power, axis=0) / total_power)
    return np.mean(bw)

def compute_spectral_flatness(mag):
    power = mag ** 2 + 1e-10
    log_power = np.log(power)
    geometric_mean = np.exp(np.mean(log_power, axis=0))
    arithmetic_mean = np.mean(power, axis=0) + 1e-10
    flatness = geometric_mean / arithmetic_mean
    return np.mean(np.clip(flatness, 0, 1))

def compute_spectral_flux(mag):
    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.mean(flux)

def compute_spectral_rolloff(freqs, mag, threshold=0.85):
    power = mag ** 2
    cumulative = np.cumsum(power, axis=0)
    total = cumulative[-1:, :] + 1e-10
    ratio = cumulative / total
    rolloff_bins = np.argmax(ratio >= threshold, axis=0)
    rolloff_hz = freqs[rolloff_bins]
    return np.mean(rolloff_hz)

def compute_spectral_skewness(freqs, mag):
    power = mag ** 2
    total_power = np.sum(power, axis=0) + 1e-10
    centroid = np.sum(freqs[:, None] * power, axis=0) / total_power
    normalized = (freqs[:, None] - centroid) / (np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * power, axis=0) / total_power) + 1e-10)
    skew = np.sum(normalized ** 3 * power, axis=0) / total_power
    return np.mean(skew)

def compute_spectral_kurtosis(freqs, mag):
    power = mag ** 2
    total_power = np.sum(power, axis=0) + 1e-10
    centroid = np.sum(freqs[:, None] * power, axis=0) / total_power
    std = np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * power, axis=0) / total_power) + 1e-10
    normalized = (freqs[:, None] - centroid) / std
    kurt = np.sum(normalized ** 4 * power, axis=0) / total_power - 3.0
    return np.mean(kurt)

def compute_spectral_slope(freqs, mag):
    mean_mag = np.mean(mag, axis=1) + 1e-10
    log_mag = 20 * np.log10(mean_mag)
    valid = freqs > 0
    x = np.log2(freqs[valid])
    y = log_mag[valid]
    slope = np.polyfit(x, y, 1)[0]
    return slope

def compute_band_energy_ratios(freqs, mag, sr=SR):
    power = np.mean(mag ** 2, axis=1)
    total = np.sum(power) + 1e-10
    edges_hz = [20, 100, 500, 2000, 6000, 20000]
    ratios = []
    for i in range(len(edges_hz) - 1):
        low_bin = np.searchsorted(freqs, edges_hz[i])
        high_bin = np.searchsorted(freqs, edges_hz[i + 1])
        band_energy = np.sum(power[low_bin:high_bin])
        ratios.append(band_energy / total)
    return np.array(ratios)

# ── RMS Energy (Tier 1 #17) ─────────────────────────────────────────────────

def compute_rms_energy(audio):
    """RMS energy in dBFS. Range: [-60, 0]."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    return np.clip(20 * np.log10(rms + 1e-10), -60, 0)

# ── Combined 81D extraction ─────────────────────────────────────────────────

def extract_metrics_80d(audio, sr=SR):
    """Extract 80D metric vector: Tier 0 (67D) + Tier 1 (13D)."""
    ltas = compute_ltas(audio, sr)
    lufs = np.array([compute_lufs(audio, sr)])
    crest = np.array([compute_crest(audio)])
    zcr = np.array([compute_zcr(audio)])

    freqs, mag = compute_stft(audio, sr)
    centroid_hz = compute_spectral_centroid(freqs, mag)

    spectral_features = np.array([
        compute_spectral_centroid(freqs, mag),
        compute_spectral_bandwidth(freqs, mag, centroid_hz),
        compute_spectral_flatness(mag),
        compute_spectral_flux(mag),
        compute_spectral_rolloff(freqs, mag),
        compute_spectral_skewness(freqs, mag),
        compute_spectral_kurtosis(freqs, mag),
        compute_spectral_slope(freqs, mag),
    ])
    band_ratios = compute_band_energy_ratios(freqs, mag, sr)
    tier1 = np.concatenate([spectral_features, band_ratios])

    return np.concatenate([ltas, lufs, crest, zcr, tier1]).astype(np.float32)

def compute_metrics_81d(audio, sr=SR):
    """Extract 81D metric vector: 80D + RMS Energy."""
    base_80d = extract_metrics_80d(audio, sr)
    rms = np.array([compute_rms_energy(audio)])
    return np.concatenate([base_80d, rms]).astype(np.float32)

print("Metric extraction: 81D (Tier 0: 67D + Tier 1: 13D + RMS: 1D)")
test_m = compute_metrics_81d(train_clips[0])
print(f"Test: shape={test_m.shape}, range=[{test_m.min():.2f}, {test_m.max():.2f}]")
print(f"  RMS value: {test_m[-1]:.2f} dBFS")
