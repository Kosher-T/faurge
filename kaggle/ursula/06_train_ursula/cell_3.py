# %% [markdown]
# ## Training Loop
#
# Main SAC training with curriculum learning, cluster masking, periodic
# logging, checkpointing, and audio rollout recording.

# ══════════════════════════════════════════════════════════════════════════════
# Inline UrsulaDSPEnv (self-contained for this notebook)
# ══════════════════════════════════════════════════════════════════════════════

import soundfile as sf
import librosa

BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024


def bark_edges(n_bands, low_hz, high_hz):
    low_bark = 13 * np.arctan(0.00076 * low_hz) + 3.5 * np.arctan((low_hz / 7500.0) ** 2)
    high_bark = 13 * np.arctan(0.00076 * high_hz) + 3.5 * np.arctan((high_hz / 7500.0) ** 2)
    edges_bark = np.linspace(low_bark, high_bark, n_bands + 1)
    edges_hz = np.zeros_like(edges_bark)
    for i, b in enumerate(edges_bark):
        lo, hi = 0.0, 100000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            bark_mid = 13 * np.arctan(0.00076 * mid) + 3.5 * np.arctan((mid / 7500.0) ** 2)
            if bark_mid < b:
                lo = mid
            else:
                hi = mid
        edges_hz[i] = (lo + hi) / 2
    return edges_hz


BARK_EDGES = bark_edges(BARK_N_BANDS, BARK_LOW_HZ, BARK_HIGH_HZ)


def compute_ltas_64(audio, sr):
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    ltas = np.zeros(BARK_N_BANDS)
    for i in range(BARK_N_BANDS):
        mask = (freqs >= BARK_EDGES[i]) & (freqs < BARK_EDGES[i + 1])
        if mask.any():
            ltas[i] = 10 * np.log10(S[mask].mean() + 1e-10)
        else:
            ltas[i] = -100.0
    return ltas


def compute_lufs(audio, sr):
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    weights = np.ones_like(freqs)
    weights[freqs < 100] = 0.5
    weights[freqs > 10000] = 0.8
    loudness = np.sum(S * weights) / len(audio)
    if loudness < 1e-10:
        return -70.0
    return -0.691 + 10 * np.log10(loudness + 1e-10)


def compute_crest_db(audio):
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms < 1e-10:
        return 0.0
    return 20 * np.log10(peak / rms)


def compute_zcr(audio):
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(audio))
    return float(crossings)


def extract_metrics_67d(audio):
    ltas = compute_ltas_64(audio, SR)
    lufs = np.array([compute_lufs(audio, SR)])
    crest = np.array([compute_crest_db(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr]).astype(np.float32)


def compute_reward(mse, floor, initial_mse):
    """Reward that provides gradient across the full MSE range.

    Uses log-scaling throughout to avoid saturation when mse >> initial_mse
    (common during warmup with random actions).

    Returns:
        +1.0 if mse <= floor (solved)
        Smooth value in (-1, +1) otherwise, always with gradient
    """
    if mse <= floor:
        return 1.0

    # ── Base penalty: log-distance from floor ──
    # log(mse/floor) compresses huge MSE range to manageable scale
    # tanh keeps it bounded; scale=0.1 → tanh reaches ~0.76 at ratio=10000
    log_ratio = np.log(mse / max(floor, 1e-6))
    base_penalty = -np.tanh(log_ratio * 0.1)  # in (-1, 0)

    # ── Progress bonus: improvement relative to start ──
    # Also log-scaled so it can't dominate/saturate
    if initial_mse > floor and mse < initial_mse:
        # How much of the log-gap have we closed?
        # 1.0 when mse=floor, 0.0 when mse=initial_mse
        log_total = np.log(initial_mse / max(floor, 1e-6))
        log_remaining = np.log(mse / max(floor, 1e-6))
        fraction_closed = 1.0 - (log_remaining / max(log_total, 1e-6))
        bonus = 0.5 * fraction_closed  # up to +0.5
    else:
        bonus = 0.0

    return float(np.clip(base_penalty + bonus, -1.0, 1.0))


def decode_action(action):
    """Decode 227D action → per-plugin config dicts (from Phase 5)."""
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class _PR:
        name: str; low: float; high: float; log: bool = False

    _eq = []
    for _b in range(31):
        _eq.extend([
            _PR(f"eq_band{_b+1}_freq", 20.0, 20000.0, log=True),
            _PR(f"eq_band{_b+1}_gain", -24.0, 24.0),
            _PR(f"eq_band{_b+1}_q", 0.1, 10.0),
            _PR(f"eq_band{_b+1}_filter_type", 0.0, 6.0),
            _PR(f"eq_band{_b+1}_stereo_skew", -6.0, 6.0),
            _PR(f"eq_band{_b+1}_dynamic_depth", 0.0, 1.0),
        ])
    _all = _eq + [
        _PR("comp_threshold", -60.0, 0.0), _PR("comp_ratio", 1.0, 20.0),
        _PR("comp_attack", 0.1, 100.0), _PR("comp_release", 10.0, 1000.0),
        _PR("comp_knee", 0.0, 12.0), _PR("comp_lookahead", 0.0, 10.0),
        _PR("comp_hold", 0.0, 200.0), _PR("comp_wet_dry", 0.0, 1.0),
        _PR("comp_stereo_link", 0.0, 1.0), _PR("comp_sidechain_hp", 20.0, 500.0),
        _PR("comp_sidechain_lp", 500.0, 20000.0, log=True),
        _PR("comp_saturate_drive", 0.0, 12.0), _PR("comp_output_trim", -12.0, 12.0),
        _PR("comp_detector_type", 0.0, 3.0),
        _PR("esser_center", 4000.0, 10000.0, log=True), _PR("esser_threshold", -60.0, 0.0),
        _PR("esser_ratio", 0.25, 20.0), _PR("esser_bandwidth", 500.0, 4000.0, log=True),
        _PR("esser_attack", 0.1, 50.0), _PR("esser_release", 10.0, 500.0),
        _PR("sat_drive", 0.0, 24.0), _PR("sat_mix", 0.0, 1.0),
        _PR("sat_type", 0.0, 3.0), _PR("sat_hpf", 20.0, 500.0),
        _PR("sat_lpf", 2000.0, 20000.0, log=True), _PR("sat_oversampling", 0.0, 3.0),
        _PR("sat_output_trim", -12.0, 12.0),
        _PR("lim_ceiling", -12.0, 0.0), _PR("lim_release", 1.0, 500.0),
        _PR("lim_lookahead", 0.0, 10.0), _PR("lim_clip_mode", 0.0, 1.0),
        _PR("lim_stereo_link", 0.0, 1.0), _PR("lim_oversampling", 0.0, 3.0),
        _PR("trans_attack_gain", -24.0, 24.0), _PR("trans_sustain_gain", -24.0, 24.0),
        _PR("trans_attack_time", 0.1, 50.0), _PR("trans_release_time", 10.0, 500.0),
        _PR("trans_sensitivity", -30.0, 0.0), _PR("trans_mix", 0.0, 1.0),
        _PR("gain_db", -12.0, 12.0), _PR("stereo_balance", -1.0, 1.0),
    ]
    lows = np.array([p.low for p in _all], dtype=np.float32)
    highs = np.array([p.high for p in _all], dtype=np.float32)
    is_log = np.array([p.log for p in _all], dtype=bool)

    _CAT = set(range(2, 186, 6))
    _CAT.add(186 + 13); _CAT.add(206 + 2); _CAT.add(206 + 5)
    _CAT.add(213 + 3); _CAT.add(213 + 5)

    vals = np.zeros_like(action)
    cont = np.array([i not in _CAT and not is_log[i] for i in range(OUTPUT_DIM)])
    vals[cont] = (action[cont] + 1.0) * 0.5 * (highs[cont] - lows[cont]) + lows[cont]
    log_m = np.array([i not in _CAT and is_log[i] for i in range(OUTPUT_DIM)])
    ll = np.log(np.maximum(lows[log_m], 1e-8)); lh = np.log(np.maximum(highs[log_m], 1e-8))
    vals[log_m] = np.exp((action[log_m] + 1.0) * 0.5 * (lh - ll) + ll)
    cat = sorted(_CAT); ca = np.array(cat)
    vals[ca] = np.clip(np.round((action[ca] + 1.0) * 0.5 * (highs[ca] - lows[ca]) + lows[ca]), lows[ca], highs[ca])

    params = {p.name: vals[i] for i, p in enumerate(_all)}
    _FT = ["peak", "low_shelf", "high_shelf", "highpass", "lowpass", "bandpass", "notch"]
    _DT = ["RMS", "peak", "feed_forward", "feed_back"]
    _ST = ["tube", "tape", "diode", "asymmetric"]
    _OS = [1, 2, 4, 8]; _CM = ["hard", "soft"]

    eq = []
    for b in range(31):
        fi = max(0, min(6, int(round(params[f"eq_band{b+1}_filter_type"]))))
        eq.append({"freq_hz": float(params[f"eq_band{b+1}_freq"]), "gain_db": float(params[f"eq_band{b+1}_gain"]),
                    "q": float(params[f"eq_band{b+1}_q"]), "filter_type": _FT[fi],
                    "stereo_skew_db": float(params[f"eq_band{b+1}_stereo_skew"]),
                    "dynamic_depth": float(params[f"eq_band{b+1}_dynamic_depth"])})
    di = max(0, min(3, int(round(params["comp_detector_type"]))))
    comp = {"threshold_db": float(params["comp_threshold"]), "ratio": float(params["comp_ratio"]),
            "attack_ms": float(params["comp_attack"]), "release_ms": float(params["comp_release"]),
            "knee_db": float(params["comp_knee"]), "lookahead_ms": float(params["comp_lookahead"]),
            "hold_ms": float(params["comp_hold"]), "wet_dry_mix": float(params["comp_wet_dry"]),
            "stereo_link": float(params["comp_stereo_link"]),
            "sidechain_hp_hz": float(params["comp_sidechain_hp"]),
            "sidechain_lp_hz": float(params["comp_sidechain_lp"]),
            "saturate_drive_db": float(params["comp_saturate_drive"]),
            "output_trim_db": float(params["comp_output_trim"]), "detector_type": _DT[di]}
    esser = {"center_freq_hz": float(params["esser_center"]), "threshold_db": float(params["esser_threshold"]),
             "ratio": float(params["esser_ratio"]), "bandwidth_hz": float(params["esser_bandwidth"]),
             "attack_ms": float(params["esser_attack"]), "release_ms": float(params["esser_release"])}
    sti = max(0, min(3, int(round(params["sat_type"]))))
    soi = max(0, min(3, int(round(params["sat_oversampling"]))))
    sat = {"drive_db": float(params["sat_drive"]), "mix": float(params["sat_mix"]),
           "sat_type": _ST[sti], "hpf_hz": float(params["sat_hpf"]),
           "lpf_hz": float(params["sat_lpf"]), "oversampling": _OS[soi],
           "output_trim_db": float(params["sat_output_trim"])}
    lci = max(0, min(1, int(round(params["lim_clip_mode"]))))
    loi = max(0, min(3, int(round(params["lim_oversampling"]))))
    lim = {"ceiling_db": float(params["lim_ceiling"]), "release_ms": float(params["lim_release"]),
           "lookahead_ms": float(params["lim_lookahead"]), "clip_mode": _CM[lci],
           "stereo_link": float(params["lim_stereo_link"]), "oversampling": _OS[loi]}
    trans = {"attack_gain_db": float(params["trans_attack_gain"]),
             "sustain_gain_db": float(params["trans_sustain_gain"]),
             "attack_time_ms": float(params["trans_attack_time"]),
             "release_time_ms": float(params["trans_release_time"]),
             "sensitivity_db": float(params["trans_sensitivity"]),
             "mix": float(params["trans_mix"])}
    g = {"gain_db": float(params["gain_db"]), "stereo_balance": float(params["stereo_balance"])}

    return {"eq": eq, "compressor": comp, "esser": esser, "saturator": sat,
            "limiter": lim, "transient": trans, "gain": g}


def apply_plugins(audio, sr, plugin_dicts):
    result = audio.copy()
    result, _ = equalizer.process(result, sr, bands=plugin_dicts['eq'])
    result, _ = compressor.process(result, sr, **plugin_dicts['compressor'])
    result, _ = esser.process(result, sr, **plugin_dicts['esser'])
    result, _ = saturator.process(result, sr, **plugin_dicts['saturator'])
    result, _ = limiter.process(result, sr, **plugin_dicts['limiter'])
    result, _ = transient.process(result, sr, **plugin_dicts['transient'])
    result, _ = gain1.process(result, sr=sr, **plugin_dicts['gain'])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# UrsulaDSPEnv (inline for notebook self-containment)
# ══════════════════════════════════════════════════════════════════════════════

class UrsulaDSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pairs_data=None, metrics_data=None, cluster_data=None,
                 max_steps=MAX_STEPS, soft_clamp_k=1.0, mode="train", max_pairs=None):
        super().__init__()
        self.pairs_data = pairs_data or PAIRS_DATA
        self.metrics_data = metrics_data or METRICS_DATA
        self.cluster_data = cluster_data or CLUSTER_DATA
        self.max_steps = max_steps
        self.soft_clamp_k = soft_clamp_k
        self.mode = mode

        meta_path = self.metrics_data / 'metadata.json'
        self._metadata = json.load(open(meta_path)) if meta_path.exists() else {}

        csv_path = self.metrics_data / 'paths.csv'
        self._all_pairs = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self._all_pairs.append(row)
        self._pairs = list(self._all_pairs)

        if max_pairs is not None and max_pairs < len(self._pairs):
            self._pairs = self._pairs[:max_pairs]

        floors_path = self.cluster_data / 'identity_floors.json'
        if floors_path.exists():
            raw = json.load(open(floors_path))
            self._identity_floors = {k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()} for k, v in raw.items()}
        else:
            self._identity_floors = {}
        self._default_floor = 0.05

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(INPUT_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(OUTPUT_DIM,), dtype=np.float32)
        self._current_audio = None; self._current_metrics = None; self._reference_metrics = None
        self._cluster_onehot = None; self._cluster_id = None; self._pair_info = None
        self._step_count = 0; self._current_mse = None; self._initial_mse = 0.0

    def set_max_pairs(self, max_pairs):
        if max_pairs >= len(self._all_pairs):
            self._pairs = list(self._all_pairs)
        else:
            self._pairs = self._all_pairs[:max_pairs]

    def _get_floor(self, cid):
        return self._identity_floors.get(f"cluster_{cid}", self._default_floor)

    def _build_onehot(self, cid):
        oh = np.zeros(N_CLUSTERS_ONEHOT, dtype=np.float32)
        if 0 <= cid < N_CLUSTERS:
            oh[cid] = 1.0
        else:
            oh[N_CLUSTERS] = 1.0
        return oh

    def _load_audio(self, path):
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if len(audio) > CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        elif len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        return audio

    def _load_metrics(self, pair_id):
        tensor = torch.load(self.metrics_data / 'pairs' / f'{pair_id}.pt', map_location='cpu', weights_only=True)
        return tensor[0].numpy(), tensor[1].numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._pairs)))
        self._pair_info = self._pairs[idx]
        pair_id = self._pair_info['pair_id']
        self._current_audio = self._load_audio(self._pair_info['degraded_path'])
        self._current_metrics, self._reference_metrics = self._load_metrics(pair_id)
        self._cluster_id = int(self._pair_info['cluster_id_reference'])
        self._cluster_onehot = self._build_onehot(self._cluster_id)
        self._step_count = 0
        self._current_mse = float(np.mean((self._current_metrics - self._reference_metrics) ** 2))
        self._initial_mse = self._current_mse
        obs = np.concatenate([self._current_metrics, self._reference_metrics, self._cluster_onehot]).astype(np.float32)
        return obs, {"pair_id": pair_id, "cluster_id": self._cluster_id,
                     "initial_mse": self._current_mse, "identity_floor": self._get_floor(self._cluster_id)}

    def step(self, action):
        self._step_count += 1
        plugin_dicts = decode_action(action)
        try:
            processed = apply_plugins(self._current_audio, SR, plugin_dicts)
        except Exception as e:
            obs = np.concatenate([self._current_metrics, self._reference_metrics, self._cluster_onehot]).astype(np.float32)
            return obs, -1.0, False, False, {"error": str(e)}
        try:
            m_result = extract_metrics_67d(processed)
        except Exception as e:
            obs = np.concatenate([self._current_metrics, self._reference_metrics, self._cluster_onehot]).astype(np.float32)
            return obs, -1.0, False, False, {"error": str(e)}
        mse = float(np.mean((m_result - self._reference_metrics) ** 2))
        floor = self._get_floor(self._cluster_id)
        reward = compute_reward(mse, floor, self._initial_mse)
        self._current_audio = processed; self._current_metrics = m_result; self._current_mse = mse
        obs = np.concatenate([self._current_metrics, self._reference_metrics, self._cluster_onehot]).astype(np.float32)
        return obs, reward, mse < floor, self._step_count >= self.max_steps, {
            "mse": mse, "identity_floor": floor, "step": self._step_count, "delta_mse": self._initial_mse - mse}


print("UrsulaDSPEnv (inline) defined")
