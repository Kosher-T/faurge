# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Metric Extraction
# ══════════════════════════════════════════════════════════════════════════════
# Reused from Phase 3. Extracts 67D vector: LTAS(64) + LUFS(1) + Crest(1) + ZCR(1)

from scipy.signal import stft, butter, sosfilt

# ── Bark Scale ────────────────────────────────────────────────────────────────

def hz_to_bark(hz: float) -> float:
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)

def bark_to_hz(bark: float) -> float:
    return 600.0 * np.sinh(bark / 7.0)

def create_bark_filterbank(n_bands=BARK_N_BANDS, low_hz=BARK_LOW_HZ,
                           high_hz=BARK_HIGH_HZ, n_fft=FFT_SIZE, sr=SR):
    bark_low = hz_to_bark(low_hz)
    bark_high = hz_to_bark(high_hz)
    bark_centers = np.linspace(bark_low, bark_high, n_bands)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i in range(n_bands):
        center = bark_centers[i]
        left_edge = bark_low if i == 0 else (bark_centers[i - 1] + center) / 2.0
        right_edge = bark_high if i == n_bands - 1 else (center + bark_centers[i + 1]) / 2.0
        center_hz = bark_to_hz(center)
        left_hz = bark_to_hz(left_edge)
        right_hz = bark_to_hz(right_edge)
        for j, f in enumerate(freqs):
            if left_hz <= f < center_hz and center_hz > left_hz:
                fb[i, j] = (f - left_hz) / (center_hz - left_hz)
            elif center_hz <= f < right_hz and right_hz > center_hz:
                fb[i, j] = (right_hz - f) / (right_hz - center_hz)
    return fb

_BARK_FILTERBANK = create_bark_filterbank()

# ── LUFS (ITU-R BS.1770-4) ───────────────────────────────────────────────────

def _design_k_weighting_filters(sr=SR):
    sos_pre = butter(1, 1500.0, btype="high", fs=sr, output="sos")
    sos_rlb = butter(2, 4000.0, btype="high", fs=sr, output="sos")
    return sos_pre, sos_rlb

_K_WEIGHTING_SOS = _design_k_weighting_filters(SR)

def compute_lufs(audio, sr=SR):
    if len(audio) < sr * 0.1:
        filtered = sosfilt(_K_WEIGHTING_SOS, audio)
        return float(-0.691 + 10.0 * np.log10(np.mean(filtered ** 2) + 1e-20))
    filtered = audio.copy()
    for sos in _K_WEIGHTING_SOS:
        filtered = sosfilt(sos, filtered)
    block_size = int(0.4 * sr)
    hop_size = int(0.1 * sr)
    n_blocks = max(1, (len(filtered) - block_size) // hop_size + 1)
    block_powers = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop_size
        end = start + block_size
        if end > len(filtered):
            break
        block_powers[i] = np.mean(filtered[start:end] ** 2)
    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated = block_powers >= abs_gate
    if not np.any(gated):
        return -70.0
    abs_mean = np.mean(block_powers[gated])
    rel_thresh = abs_mean * (10.0 ** (-10.0 / 10.0))
    rel_gated = block_powers >= rel_thresh
    if not np.any(rel_gated):
        return -70.0
    return float(-0.691 + 10.0 * np.log10(np.mean(block_powers[rel_gated]) + 1e-20))

# ── Crest Factor & ZCR ───────────────────────────────────────────────────────

def compute_crest_factor(audio):
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return 20.0 * np.log10(peak / (rms + 1e-10))

def compute_zcr(audio):
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (2.0 * (len(audio) - 1)))

# ── LTAS (64D) ───────────────────────────────────────────────────────────────

def compute_ltas(audio, sr=SR):
    _, _, Zxx = stft(audio, fs=sr, nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP_SIZE,
                     window="hann", return_onesided=True)
    power = np.abs(Zxx) ** 2
    band_energy = _BARK_FILTERBANK @ power
    mean_energy = np.mean(band_energy, axis=1)
    return (10.0 * np.log10(mean_energy + 1e-10)).astype(np.float64)

# ── Main: Extract 67D ────────────────────────────────────────────────────────

def extract_metrics_67d(audio, sr=SR):
    ltas = compute_ltas(audio, sr)
    lufs = np.array([compute_lufs(audio, sr)])
    crest = np.array([compute_crest_factor(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr])

print("Metric extraction defined: extract_metrics_67d → 67D (LTAS64 + LUFS1 + Crest1 + ZCR1)")
