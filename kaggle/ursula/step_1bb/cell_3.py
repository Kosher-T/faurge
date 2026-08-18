# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Metric Extraction (67D)
# ══════════════════════════════════════════════════════════════════════════════

import scipy.signal

def compute_ltas(audio, sr=SR, fft_size=FFT_SIZE, hop_size=HOP_SIZE, n_bands=LTAS_DIM):
    freqs, times, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=fft_size, noverlap=fft_size - hop_size)
    power = np.abs(Zxx) ** 2
    low_bin = int(BARK_LOW_HZ / (sr / fft_size))
    high_bin = int(BARK_HIGH_HZ / (sr / fft_size))
    low_bin = max(low_bin, 1)
    high_bin = min(high_bin, power.shape[0])
    power = power[low_bin:high_bin, :]
    band_edges = np.linspace(0, power.shape[0], n_bands + 1, dtype=int)
    ltas = np.zeros(n_bands)
    for i in range(n_bands):
        band_power = power[band_edges[i]:band_edges[i + 1], :]
        ltas[i] = np.mean(band_power) if band_power.size > 0 else 1e-10
    ltas = np.log10(ltas + 1e-10)
    return ltas

def compute_lufs(audio, sr=SR):
    lufs = -0.691 + 10 * np.log10(np.mean(audio ** 2) + 1e-10)
    return np.clip(lufs, -70, 0)

def compute_crest(audio, frame_size=2048, hop_size=512):
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, hop_size)]
    if not frames:
        return 0.0
    rms = np.sqrt(np.mean(np.array(frames) ** 2, axis=1) + 1e-10)
    peak = np.max(np.abs(np.array(frames)), axis=1) + 1e-10
    crest_db = 20 * np.log10(peak / rms + 1e-10)
    return np.mean(crest_db)

def compute_zcr(audio, frame_size=2048, hop_size=512):
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, hop_size)]
    if not frames:
        return 0.0
    zcr = np.sum(np.abs(np.diff(np.sign(np.array(frames)))), axis=1) / (2 * frame_size)
    return np.mean(zcr)

def extract_metrics_67d(audio, sr=SR):
    ltas = compute_ltas(audio, sr)
    lufs = np.array([compute_lufs(audio, sr)])
    crest = np.array([compute_crest(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr]).astype(np.float32)

print(f"Metric extraction: 67D (LTAS64 + LUFS + Crest + ZCR)")
test_m = extract_metrics_67d(clean_clips[0])
print(f"Test: shape={test_m.shape}, range=[{test_m.min():.2f}, {test_m.max():.2f}]")
