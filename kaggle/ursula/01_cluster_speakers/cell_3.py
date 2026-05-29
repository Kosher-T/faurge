# ─── Bark Scale Utilities ────────────────────────────────────────────

def hz_to_bark(hz: float) -> float:
    """Convert Hz to Bark scale (Zwicker / Trapezoidal)."""
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def bark_to_hz(bark: float) -> float:
    """Convert Bark scale back to Hz."""
    return 600.0 * np.sinh(bark / 7.0)


def create_bark_filterbank(
    n_bands: int = BARK_N_BANDS,
    low_hz: float = BARK_LOW_HZ,
    high_hz: float = BARK_HIGH_HZ,
    n_fft: int = FFT_SIZE,
    sr: int = SR,
) -> np.ndarray:
    """
    Create triangular Bark-scale filterbank.
    Returns (n_bands, n_fft//2+1) weight matrix.
    """
    bark_low = hz_to_bark(low_hz)
    bark_high = hz_to_bark(high_hz)
    bark_centers = np.linspace(bark_low, bark_high, n_bands)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # (n_fft//2+1,)

    filterbank = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i in range(n_bands):
        center = bark_centers[i]
        if i == 0:
            left_edge = bark_low
        else:
            left_edge = (bark_centers[i - 1] + center) / 2.0
        if i == n_bands - 1:
            right_edge = bark_high
        else:
            right_edge = (center + bark_centers[i + 1]) / 2.0

        center_hz = bark_to_hz(center)
        left_hz = bark_to_hz(left_edge)
        right_hz = bark_to_hz(right_edge)

        for j, f in enumerate(freqs):
            if f < left_hz:
                filterbank[i, j] = 0.0
            elif left_hz <= f < center_hz and center_hz > left_hz:
                filterbank[i, j] = (f - left_hz) / (center_hz - left_hz)
            elif center_hz <= f < right_hz and right_hz > center_hz:
                filterbank[i, j] = (right_hz - f) / (right_hz - center_hz)
            elif f >= right_hz:
                filterbank[i, j] = 0.0
            else:
                filterbank[i, j] = 0.0

    return filterbank


# Pre-compute filterbank once
_BARK_FILTERBANK = create_bark_filterbank()

# ─── LUFS (ITU-R BS.1770-4) ─────────────────────────────────────────

def _design_k_weighting_filters(sr: int = SR):
    """
    Design the two stages of K-weighting filters per ITU-R BS.1770-4.
    Returns (sos_pre, sos_rlb) as second-order-sections.
    """
    from scipy.signal import butter

    sos_pre = butter(1, 1500.0, btype="high", fs=sr, output="sos")

    f0 = 4000.0
    gain_db = 3.0
    gain_lin = 10.0 ** (gain_db / 20.0)
    sos_rlb = butter(2, f0, btype="high", fs=sr, output="sos")

    return sos_pre, sos_rlb


def compute_lufs_1d(audio: np.ndarray, sr: int = SR) -> float:
    """
    Compute integrated LUFS per ITU-R BS.1770-4.
    Two-pass gating algorithm: absolute gate at -70 LUFS, relative gate at -10 dB.
    """
    from scipy.signal import sosfilt

    if len(audio) < sr * 0.1:
        filtered = sosfilt(_K_WEIGHTING_SOS, audio)
        mean_sq = np.mean(filtered ** 2)
        lufs = -0.691 + 10.0 * np.log10(mean_sq + 1e-20)
        return float(lufs)

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
        block = filtered[start:end]
        block_powers[i] = np.mean(block ** 2)

    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated_mask = block_powers >= abs_gate

    if not np.any(gated_mask):
        return -70.0

    abs_gated_mean = np.mean(block_powers[gated_mask])
    rel_threshold = abs_gated_mean * (10.0 ** (-10.0 / 10.0))
    rel_gated_mask = block_powers >= rel_threshold

    if not np.any(rel_gated_mask):
        return -70.0

    final_mean = np.mean(block_powers[rel_gated_mask])
    lufs = -0.691 + 10.0 * np.log10(final_mean + 1e-20)
    return float(lufs)


# Design K-weighting filters once
_K_WEIGHTING_SOS = _design_k_weighting_filters(SR)

# ─── Crest Factor & ZCR ──────────────────────────────────────────────

def compute_crest_factor(audio: np.ndarray) -> float:
    """Crest Factor in dB: 20*log10(peak/rms)."""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return 20.0 * np.log10(peak / (rms + 1e-10))


def compute_zcr(audio: np.ndarray) -> float:
    """Zero-Crossing Rate."""
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (2.0 * (len(audio) - 1)))

# ─── 67D Metrics Extractor ───────────────────────────────────────────

def compute_ltas_64d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Compute LTAS: 64D Bark-scale spectral envelope.
    STFT → power spectrum → Bark filterbank → dB → time average.
    """
    from scipy.signal import stft

    f, t, Zxx = stft(audio, fs=sr, nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP_SIZE,
                     window="hann", return_onesided=True)
    power = np.abs(Zxx) ** 2

    band_energy = _BARK_FILTERBANK @ power
    mean_energy = np.mean(band_energy, axis=1)
    ltas = 10.0 * np.log10(mean_energy + 1e-10)
    return ltas.astype(np.float64)


def compute_ltas_64d_batch(
    clips: list[np.ndarray],
    sr: int = SR,
    device: str = "cpu",
) -> list[np.ndarray]:
    """
    GPU-accelerated batch LTAS: process multiple clips in one STFT + filterbank pass.
    Falls back to CPU if torch unavailable.
    """
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        n_clips = len(clips)
        n_freqs = FFT_SIZE // 2 + 1

        clips_t = torch.tensor(np.stack(clips), dtype=torch.float32, device=device)
        fb_t = torch.tensor(_BARK_FILTERBANK, dtype=torch.float32, device=device)

        window = torch.hann_window(FFT_SIZE, device=device)
        stft_out = torch.stft(
            clips_t, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            win_length=FFT_SIZE, window=window,
            center=False, return_complex=True,
        )

        power = stft_out.abs() ** 2

        band_energy = torch.einsum("bf,cft->cbt", fb_t, power)
        mean_energy = band_energy.mean(dim=2)
        ltas = 10.0 * torch.log10(mean_energy + 1e-10)

        result = ltas.cpu().numpy().astype(np.float64)

        del clips_t, stft_out, power, band_energy, mean_energy, ltas
        if device == "cuda":
            torch.cuda.empty_cache()
        return [result[i] for i in range(n_clips)]

    except Exception:
        return [compute_ltas_64d(c, sr) for c in clips]


def extract_metrics_67d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract full 67D metric vector: LTAS(64) + LUFS(1) + Crest(1) + ZCR(1).
    """
    ltas = compute_ltas_64d(audio, sr)
    lufs = np.array([compute_lufs_1d(audio, sr)])
    crest = np.array([compute_crest_factor(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr])

# ─── Speaker Discovery ───────────────────────────────────────────────

def discover_speakers(paths: dict) -> dict[str, list[Path]]:
    """
    Discover all speakers across VCTK, LJSpeech, DAPS.
    Returns {speaker_id: [Path, ...]}.
    """
    speakers = defaultdict(list)

    vctk_root = paths["vctk"]
    if vctk_root.exists():
        for speaker_dir in sorted(vctk_root.iterdir()):
            if speaker_dir.is_dir():
                wav_files = sorted(speaker_dir.glob("*.wav"))
                if wav_files:
                    speakers[speaker_dir.name].extend(wav_files)

    ljspeech_root = paths["ljspeech"]
    if ljspeech_root.exists():
        wav_files = sorted(ljspeech_root.glob("*.wav"))
        if wav_files:
            speakers["LJ001"].extend(wav_files)

    daps_root = paths["daps"]
    if daps_root.exists():
        for wav_file in sorted(daps_root.rglob("*.wav")):
            speaker_id = wav_file.stem.split("_")[0]
            speakers[speaker_id].append(wav_file)

    return dict(speakers)

# ─── Speaker Profile Computation ─────────────────────────────────────

def _extract_clip_metrics(audio: np.ndarray) -> np.ndarray:
    """Extract 67D metrics from a single clip (already at SR, mono, CLIP_SAMPLES)."""
    return extract_metrics_67d(audio, SR)


def compute_speaker_profiles(
    speakers: dict[str, list[Path]],
    samples_per_speaker: int = SAMPLES_PER_SPEAKER,
) -> dict[str, np.ndarray]:
    """
    Memory-efficient streaming profile computation.
    Never holds more than one clip in memory at a time.
    """
    import gc
    profiles = {}
    total_speakers = len(speakers)

    for idx, (speaker_id, clip_paths) in enumerate(speakers.items(), 1):
        t0 = time.time()

        total_segments = 0
        for p in clip_paths:
            try:
                info = sf.info(str(p))
                duration_samples = int(info.duration * SR)
                if duration_samples < SR * 0.5:
                    continue
                if info.duration <= CLIP_SEC + 0.5:
                    total_segments += 1
                else:
                    total_segments += max(1, duration_samples // CLIP_SAMPLES)
            except Exception:
                continue

        if total_segments == 0:
            print(f"  [{idx}/{total_speakers}] {speaker_id}: no valid segments, skipping")
            continue

        n_sample = min(samples_per_speaker, total_segments)
        sample_indices = set(sorted(random.sample(range(total_segments), n_sample)))

        metrics_list = []
        seg_idx = 0
        for p in clip_paths:
            if len(metrics_list) >= n_sample:
                break
            try:
                audio, file_sr = sf.read(p, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if file_sr != SR:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
            except Exception:
                seg_idx += max(1, int(sf.info(str(p)).duration * SR) // CLIP_SAMPLES)
                continue

            if len(audio) < SR * 0.5:
                seg_idx += 1
                del audio
                continue

            if len(audio) <= CLIP_SAMPLES:
                if seg_idx in sample_indices:
                    chunk = audio
                    if len(chunk) < CLIP_SAMPLES:
                        chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms > 1e-4:
                        metrics_list.append(_extract_clip_metrics(chunk))
                seg_idx += 1
                del audio
            else:
                for start in range(0, len(audio) - SR, CLIP_SAMPLES):
                    if seg_idx in sample_indices:
                        chunk = audio[start : start + CLIP_SAMPLES]
                        if len(chunk) < CLIP_SAMPLES:
                            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
                        rms = np.sqrt(np.mean(chunk ** 2))
                        if rms > 1e-4:
                            metrics_list.append(_extract_clip_metrics(chunk))
                    seg_idx += 1
                    if len(metrics_list) >= n_sample:
                        break
                del audio

        if metrics_list:
            profile = np.mean(metrics_list, axis=0)
            profiles[speaker_id] = profile

        elapsed = time.time() - t0
        print(f"  [{idx}/{total_speakers}] {speaker_id}: "
              f"{len(metrics_list)} clips → 67D ({elapsed:.1f}s)")

        gc.collect()

    return profiles

# ─── Profile Cache ────────────────────────────────────────────────────

def save_profiles(profiles: dict[str, np.ndarray], path: Path):
    """Save speaker profiles to .npz for fast reload."""
    np.savez(path, **profiles)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(profiles)} profiles → {path} ({size_mb:.1f} MB)")


def load_profiles(path: Path) -> dict[str, np.ndarray]:
    """Load cached speaker profiles."""
    data = np.load(path)
    profiles = {key: data[key] for key in data.files}
    print(f"Loaded {len(profiles)} profiles from cache ({path})")
    return profiles

# ─── Clustering ───────────────────────────────────────────────────────

def run_clustering(
    speaker_profiles: dict[str, np.ndarray],
    n_clusters: int = N_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    K-means clustering on speaker profiles.
    Returns (labels, centroids_67d_dict, thresholds_dict, speaker_ids, X, X_scaled).
    """
    speaker_ids = sorted(speaker_profiles.keys())
    X = np.array([speaker_profiles[sid] for sid in speaker_ids])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n── Silhouette Score Comparison ──")
    for k in [4, 6, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels_k)
        print(f"  k={k}: silhouette = {score:.4f}")

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)

    centroids_67d_dict = {}
    thresholds_dict = {}

    for c in range(n_clusters):
        mask = labels == c
        centroid_67d = np.mean(X[mask], axis=0)

        cluster_profiles = X[mask]
        if len(cluster_profiles) > 1:
            pairwise_mse = []
            for i in range(len(cluster_profiles)):
                for j in range(i + 1, len(cluster_profiles)):
                    mse = np.mean((cluster_profiles[i] - cluster_profiles[j]) ** 2)
                    pairwise_mse.append(mse)
            threshold = float(np.percentile(pairwise_mse, 95))
        else:
            threshold = 0.0

        centroids_67d_dict[f"cluster_{c}"] = centroid_67d
        thresholds_dict[f"cluster_{c}"] = threshold

    print(f"\n── Cluster Assignments ──")
    for c in range(n_clusters):
        members = [speaker_ids[i] for i in range(len(labels)) if labels[i] == c]
        print(f"  Cluster {c} ({len(members)} speakers): "
              f"{', '.join(members[:10])}{'...' if len(members) > 10 else ''}")

    return labels, centroids_67d_dict, thresholds_dict, speaker_ids, X, X_scaled

# ─── Identity Floors ─────────────────────────────────────────────────

def compute_identity_floors(
    speaker_profiles: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    speaker_ids: list[str],
) -> dict:
    """
    Compute within-cluster and cross-cluster pairwise MSE identity floors.
    """
    X = np.array([speaker_profiles[sid] for sid in speaker_ids])
    n_clusters = len(set(cluster_labels))
    floors = {}

    print("\n── Within-Cluster Identity Floors ──")
    for c in range(n_clusters):
        mask = np.where(cluster_labels == c)[0]
        if len(mask) < 2:
            floors[f"cluster_{c}"] = 0.0
            continue

        cluster_profiles = X[mask]
        pairwise_mse = []
        for i in range(len(cluster_profiles)):
            for j in range(i + 1, len(cluster_profiles)):
                mse = np.mean((cluster_profiles[i] - cluster_profiles[j]) ** 2)
                pairwise_mse.append(mse)
        avg_mse = float(np.mean(pairwise_mse))
        floors[f"cluster_{c}"] = avg_mse
        print(f"  Cluster {c}: {avg_mse:.6f} (n={len(mask)}, pairs={len(pairwise_mse)})")

    floors["cross"] = {}
    print("\n── Cross-Cluster Identity Floors ──")
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            mask1 = np.where(cluster_labels == c1)[0]
            mask2 = np.where(cluster_labels == c2)[0]
            if len(mask1) == 0 or len(mask2) == 0:
                continue

            profiles1 = X[mask1]
            profiles2 = X[mask2]
            cross_mse = []
            for p1 in profiles1:
                for p2 in profiles2:
                    mse = float(np.mean((p1 - p2) ** 2))
                    cross_mse.append(mse)
            avg_cross = float(np.mean(cross_mse))
            key = f"cluster_{c1}_cluster_{c2}"
            floors["cross"][key] = avg_cross
            print(f"  {key}: {avg_cross:.6f}")

    return floors

# ─── Validation ───────────────────────────────────────────────────────

def validate_results(
    cluster_labels: np.ndarray,
    identity_floors: dict,
    centroids_67d: dict,
    thresholds: dict,
):
    """Run sanity checks on clustering results."""
    print("\n═══════════════════════════════════════════")
    print("  VALIDATION")
    print("═══════════════════════════════════════════")
    n_clusters = len(set(cluster_labels))
    all_pass = True

    within_vals = [identity_floors[f"cluster_{c}"] for c in range(n_clusters)
                   if f"cluster_{c}" in identity_floors]
    cross_vals = list(identity_floors.get("cross", {}).values())
    if within_vals and cross_vals:
        avg_within = np.mean(within_vals)
        avg_cross = np.mean(cross_vals)
        check = avg_within < avg_cross
        status = "PASS" if check else "FAIL"
        if not check:
            all_pass = False
        print(f"  [{status}] Within-cluster avg ({avg_within:.6f}) < "
              f"cross-cluster avg ({avg_cross:.6f})")
    else:
        print("  [SKIP] Not enough data for floor comparison")

    all_positive = all(v > 0 for v in within_vals) and all(v > 0 for v in cross_vals)
    status = "PASS" if all_positive else "FAIL"
    if not all_positive:
        all_pass = False
    print(f"  [{status}] All identity floors positive")

    centroid_vecs = np.array([centroids_67d[f"cluster_{c}"] for c in range(n_clusters)])
    centroid_mses = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mse = np.mean((centroid_vecs[i] - centroid_vecs[j]) ** 2)
            centroid_mses.append(mse)
    if centroid_mses and within_vals:
        avg_centroid_mse = np.mean(centroid_mses)
        avg_within = np.mean(within_vals)
        ratio = avg_centroid_mse / (avg_within + 1e-10)
        check = ratio > 1.5
        status = "PASS" if check else "WARN"
        if not check:
            all_pass = False
        print(f"  [{status}] Avg centroid separation ({avg_centroid_mse:.2f}) / "
              f"avg within-cluster ({avg_within:.2f}) = {ratio:.2f}x")
    else:
        print("  [SKIP] Not enough data for centroid separation check")

    if centroid_mses:
        min_centroid_dist = min(centroid_mses)
        threshold_vals = [t for t in thresholds.values() if t > 0]
        if threshold_vals:
            thresholds_valid = all(t < min_centroid_dist for t in threshold_vals)
            status = "PASS" if thresholds_valid else "WARN"
            print(f"  [{status}] {len(threshold_vals)} thresholds in (0, "
                  f"{min_centroid_dist:.2f}) — {sum(1 for t in thresholds.values() if t == 0)} skipped (n<2)")
        else:
            print("  [SKIP] All thresholds 0 (clusters too small)")
    else:
        print("  [SKIP] Not enough data for threshold check")

    sizes = [int(np.sum(cluster_labels == c)) for c in range(n_clusters)]
    min_size = min(sizes)
    max_size = max(sizes)
    balance_ratio = max_size / (min_size + 1e-10)
    check = min_size >= 3
    status = "PASS" if check else "WARN"
    if not check:
        all_pass = False
    print(f"  [{status}] Cluster sizes: min={min_size}, max={max_size}, "
          f"balance={balance_ratio:.1f}x")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

# ─── Save Outputs ─────────────────────────────────────────────────────

def save_outputs(
    speaker_profiles: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    centroids_67d: dict,
    thresholds: dict,
    identity_floors: dict,
    speaker_ids: list[str],
    output_dir: Path,
):
    """Save all artifacts to JSON files."""
    clusters = {}
    for idx, sid in enumerate(speaker_ids):
        clusters[sid] = {
            "cluster": int(cluster_labels[idx]),
            "profile_67d": speaker_profiles[sid].tolist(),
            "gender": "unknown",
        }
    clusters_path = output_dir / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"\nSaved {clusters_path} ({clusters_path.stat().st_size / 1024:.1f} KB)")

    centroids_json = {}
    for key in centroids_67d:
        centroids_json[key] = {
            "centroid_67d": centroids_67d[key].tolist(),
            "threshold": thresholds[key],
        }
    centroids_path = output_dir / "cluster_centroids.json"
    with open(centroids_path, "w") as f:
        json.dump(centroids_json, f, indent=2)
    print(f"Saved {centroids_path} ({centroids_path.stat().st_size / 1024:.1f} KB)")

    floors_path = output_dir / "identity_floors.json"
    with open(floors_path, "w") as f:
        json.dump(identity_floors, f, indent=2)
    print(f"Saved {floors_path} ({floors_path.stat().st_size / 1024:.1f} KB)")
