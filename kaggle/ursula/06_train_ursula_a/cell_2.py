# %% [markdown]
# ## Generate Supervised Targets via Inverse Degradation
#
# For each degraded-reference pair, compute the approximate inverse of the
# degradation parameters that were applied. This gives physically meaningful
# training targets instead of random search noise.
#
# Inversion strategy:
# - EQ: negate gain_db (exact inverse), keep freq/Q/filter_type
# - Compressor: disable (threshold=0, ratio=1, wet_dry=0)
# - Esser: disable (threshold=0)
# - Saturator: disable (drive=0, mix=0)
# - Limiter: disable (ceiling=0)
# - Transient: negate gains or disable (mix=0)
# - Gain: negate gain_db

import soundfile as sf
import librosa

# ── Metrics extraction (same as Phase 5/6) ──

BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096

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
            if bark_mid < b: lo = mid
            else: hi = mid
        edges_hz[i] = (lo + hi) / 2
    return edges_hz

BARK_EDGES = bark_edges(BARK_N_BANDS, BARK_LOW_HZ, BARK_HIGH_HZ)

def compute_ltas_64(audio, sr):
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    ltas = np.zeros(BARK_N_BANDS)
    for i in range(BARK_N_BANDS):
        mask = (freqs >= BARK_EDGES[i]) & (freqs < BARK_EDGES[i + 1])
        ltas[i] = 10 * np.log10(S[mask].mean() + 1e-10) if mask.any() else -100.0
    return ltas

def compute_lufs(audio, sr):
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    w = np.ones_like(freqs); w[freqs < 100] = 0.5; w[freqs > 10000] = 0.8
    loudness = np.sum(S * w) / len(audio)
    return -0.691 + 10 * np.log10(loudness + 1e-10) if loudness > 1e-10 else -70.0

def compute_crest_db(audio):
    peak = np.max(np.abs(audio)); rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    return 20 * np.log10(peak / rms) if rms > 1e-10 else 0.0

def compute_zcr(audio):
    signs = np.sign(audio); signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs))) / (2 * len(audio)))

def extract_metrics_67d(audio):
    return np.concatenate([
        compute_ltas_64(audio, SR),
        np.array([compute_lufs(audio, SR)]),
        np.array([compute_crest_db(audio)]),
        np.array([compute_zcr(audio)]),
    ]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Decode + apply plugins (from Phase 6 cell_3)
# ══════════════════════════════════════════════════════════════════════════════

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
_ALL = _eq + [
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
_LOWS = np.array([p.low for p in _ALL], dtype=np.float32)
_HIGHS = np.array([p.high for p in _ALL], dtype=np.float32)
_IS_LOG = np.array([p.log for p in _ALL], dtype=bool)
_CAT = set(range(2, 186, 6))
_CAT.update([186+13, 206+2, 206+5, 213+3, 213+5])


def decode_action(action):
    vals = np.zeros_like(action)
    cont = np.array([i not in _CAT and not _IS_LOG[i] for i in range(OUTPUT_DIM)])
    vals[cont] = (action[cont] + 1.0) * 0.5 * (_HIGHS[cont] - _LOWS[cont]) + _LOWS[cont]
    log_m = np.array([i not in _CAT and _IS_LOG[i] for i in range(OUTPUT_DIM)])
    ll = np.log(np.maximum(_LOWS[log_m], 1e-8)); lh = np.log(np.maximum(_HIGHS[log_m], 1e-8))
    vals[log_m] = np.exp((action[log_m] + 1.0) * 0.5 * (lh - ll) + ll)
    ca = sorted(_CAT); ca_a = np.array(ca)
    vals[ca_a] = np.clip(np.round((action[ca_a] + 1.0) * 0.5 * (_HIGHS[ca_a] - _LOWS[ca_a]) + _LOWS[ca_a]), _LOWS[ca_a], _HIGHS[ca_a])
    params = {p.name: vals[i] for i, p in enumerate(_ALL)}
    _FT = ["peak","low_shelf","high_shelf","highpass","lowpass","bandpass","notch"]
    _DT = ["RMS","peak","feed_forward","feed_back"]
    _ST = ["tube","tape","diode","asymmetric"]
    _OS = [1,2,4,8]; _CM = ["hard","soft"]
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
# Inverse Degradation: compute approximate restoration action from degradation params
# ══════════════════════════════════════════════════════════════════════════════

def _inv_linear(val, low, high):
    """Encode a linear param to [-1, 1]."""
    return float(np.clip((val - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0))

def _inv_log(val, low, high):
    """Encode a log-scale param to [-1, 1]."""
    val = max(val, 1e-8)
    log_low = math.log(max(low, 1e-8))
    log_high = math.log(max(high, 1e-8))
    return float(np.clip((math.log(val) - log_low) / (log_high - log_low) * 2.0 - 1.0, -1.0, 1.0))

def _inv_cat(val, low, high):
    """Encode a categorical param to [-1, 1]."""
    return float(np.clip((val - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0))


def compute_inverse_action(deg_params):
    """Convert degradation parameters to an approximate inverse action vector.

    Strategy:
    - EQ: negate gain_db (exact inverse for linear EQ)
    - Compressor: disable (threshold=0, ratio=1, wet_dry=0)
    - Esser: disable (threshold=0)
    - Saturator: disable (drive=0, mix=0)
    - Limiter: disable (ceiling=0)
    - Transient: negate gains
    - Gain: negate gain_db
    """
    inv = np.zeros(OUTPUT_DIM, dtype=np.float32)

    # ── EQ: 31 bands × 6 params = 186D ──
    # degradation uses 1-6 random bands; unused bands have gain=0 (identity)
    deg_bands = deg_params.get('eq_bands', [])
    for b in range(31):
        idx = b * 6
        if b < len(deg_bands):
            band = deg_bands[b]
            freq = band.get('freq_hz', 1000.0)
            gain = band.get('gain_db', 0.0)
            q = band.get('q', 1.0)
            ft_str = band.get('filter_type', 'peak')
            stereo_skew = band.get('stereo_skew_db', 0.0)
            dyn_depth = band.get('dynamic_depth', 0.0)

            _FT_MAP = {"peak": 0, "low_shelf": 1, "high_shelf": 2,
                       "highpass": 3, "lowpass": 4, "bandpass": 5, "notch": 6}
            ft_val = _FT_MAP.get(ft_str, 0)
        else:
            freq, gain, q = 1000.0, 0.0, 1.0
            ft_val, stereo_skew, dyn_depth = 0, 0.0, 0.0

        inv[idx + 0] = _inv_log(freq, 20.0, 20000.0)           # freq: keep same
        inv[idx + 1] = _inv_linear(-gain, -24.0, 24.0)         # gain: NEGATE
        inv[idx + 2] = _inv_linear(q, 0.1, 10.0)               # q: keep same
        inv[idx + 3] = _inv_cat(ft_val, 0.0, 6.0)              # filter_type: keep same
        inv[idx + 4] = _inv_linear(0.0, -6.0, 6.0)             # stereo_skew: zero (no info to invert)
        inv[idx + 5] = _inv_linear(0.0, 0.0, 1.0)              # dynamic_depth: zero

    # ── Compressor: 14D (186-199) — disable ──
    comp = deg_params.get('compressor', {})
    inv[186] = _inv_linear(0.0, -60.0, 0.0)        # threshold: 0 dB (max = no compression)
    inv[187] = _inv_linear(1.0, 1.0, 20.0)         # ratio: 1:1 (unity = no compression)
    inv[188] = _inv_linear(comp.get('attack_ms', 10.0), 0.1, 100.0)   # attack: keep same
    inv[189] = _inv_linear(comp.get('release_ms', 100.0), 10.0, 1000.0)  # release: keep same
    inv[190] = _inv_linear(comp.get('knee_db', 6.0), 0.0, 12.0)      # knee: keep same
    inv[191] = _inv_linear(comp.get('lookahead_ms', 0.0), 0.0, 10.0) # lookahead: keep same
    inv[192] = _inv_linear(comp.get('hold_ms', 0.0), 0.0, 200.0)     # hold: keep same
    inv[193] = _inv_linear(0.0, 0.0, 1.0)          # wet_dry: 0 (fully dry = bypass)
    inv[194] = _inv_linear(comp.get('stereo_link', 0.0), 0.0, 1.0)   # stereo_link: keep same
    inv[195] = _inv_linear(comp.get('sidechain_hp_hz', 20.0), 20.0, 500.0)
    inv[196] = _inv_log(comp.get('sidechain_lp_hz', 20000.0), 500.0, 20000.0)
    inv[197] = _inv_linear(comp.get('saturate_drive_db', 0.0), 0.0, 12.0)
    inv[198] = _inv_linear(comp.get('output_trim_db', 0.0), -12.0, 12.0)
    inv[199] = _inv_cat(comp.get('detector_type', 0), 0.0, 3.0)  # detector_type: keep same

    # ── Esser: 6D (200-205) — disable ──
    ess = deg_params.get('esser', {})
    inv[200] = _inv_log(ess.get('center_freq_hz', 6000.0), 4000.0, 10000.0)
    inv[201] = _inv_linear(0.0, -60.0, 0.0)        # threshold: 0 (max = no reduction)
    inv[202] = _inv_linear(1.0, 0.25, 20.0)        # ratio: 1:1 (unity = no reduction)
    inv[203] = _inv_log(ess.get('bandwidth_hz', 2000.0), 500.0, 4000.0)
    inv[204] = _inv_linear(ess.get('attack_ms', 5.0), 0.1, 50.0)
    inv[205] = _inv_linear(ess.get('release_ms', 50.0), 10.0, 500.0)

    # ── Saturator: 7D (206-212) — disable ──
    sat = deg_params.get('saturator', {})
    inv[206] = _inv_linear(0.0, 0.0, 24.0)         # drive: 0 (no saturation)
    inv[207] = _inv_linear(0.0, 0.0, 1.0)          # mix: 0 (fully dry)
    inv[208] = _inv_cat(0.0, 0.0, 3.0)             # type: tube (keep same)
    inv[209] = _inv_linear(sat.get('hpf_hz', 80.0), 20.0, 500.0)
    inv[210] = _inv_log(sat.get('lpf_hz', 18000.0), 2000.0, 20000.0)
    inv[211] = _inv_cat(1.0, 0.0, 3.0)             # oversampling: 1x
    inv[212] = _inv_linear(sat.get('output_trim_db', 0.0), -12.0, 12.0)

    # ── Limiter: 6D (213-218) — disable ──
    lim = deg_params.get('limiter', {})
    inv[213] = _inv_linear(0.0, -12.0, 0.0)        # ceiling: 0 dB (max = no limiting)
    inv[214] = _inv_linear(lim.get('release_ms', 50.0), 1.0, 500.0)
    inv[215] = _inv_linear(lim.get('lookahead_ms', 0.0), 0.0, 10.0)
    inv[216] = _inv_cat(0.0, 0.0, 1.0)             # clip_mode: hard (keep same)
    inv[217] = _inv_linear(lim.get('stereo_link', 0.0), 0.0, 1.0)
    inv[218] = _inv_cat(1.0, 0.0, 3.0)             # oversampling: 1x

    # ── Transient: 6D (219-224) — negate gains ──
    trans = deg_params.get('transient', {})
    inv[219] = _inv_linear(-trans.get('attack_gain_db', 0.0), -24.0, 24.0)   # negate
    inv[220] = _inv_linear(-trans.get('sustain_gain_db', 0.0), -24.0, 24.0)  # negate
    inv[221] = _inv_linear(trans.get('attack_time_ms', 5.0), 0.1, 50.0)
    inv[222] = _inv_linear(trans.get('release_time_ms', 50.0), 10.0, 500.0)
    inv[223] = _inv_linear(trans.get('sensitivity_db', -15.0), -30.0, 0.0)
    inv[224] = _inv_linear(trans.get('mix', 0.5), 0.0, 1.0)

    # ── Gain: 2D (225-226) — negate ──
    g = deg_params.get('gain', {})
    inv[225] = _inv_linear(-g.get('gain_db', 0.0), -12.0, 12.0)   # negate
    inv[226] = _inv_linear(-g.get('stereo_balance', 0.0), -1.0, 1.0)  # negate

    return np.clip(inv, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Load pairs and audio
# ══════════════════════════════════════════════════════════════════════════════

csv_path = METRICS_DATA / 'paths.csv'
all_pairs = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        all_pairs.append(row)

# Load degradation params
deg_params_path = PAIRS_DATA / 'degradation_params.json'
if not deg_params_path.exists():
    # Fallback: try alternative locations
    for alt in [PAIRS_DATA / 'degradation_params.json',
                INPUT / 'notebooks/itorousa/02-generate-degraded-pairs/ursula_dataset/degradation_params.json']:
        if alt.exists():
            deg_params_path = alt
            break

if deg_params_path.exists():
    with open(deg_params_path) as f:
        all_deg_params = json.load(f)
    print(f"Loaded degradation params for {len(all_deg_params)} pairs")
else:
    raise FileNotFoundError(
        f"degradation_params.json not found. Checked:\n"
        f"  {PAIRS_DATA / 'degradation_params.json'}\n"
        f"  Searched: {deg_params_path}"
    )

# Filter to pairs that have degradation params
pairs_with_params = [p for p in all_pairs if p['pair_id'] in all_deg_params]
if MAX_PAIRS is not None:
    pairs_with_params = pairs_with_params[:MAX_PAIRS]
print(f"Using {len(pairs_with_params)} pairs (from {len(all_pairs)} total)")

# Load audio and metrics for each pair
pair_data = []
for p in pairs_with_params:
    pair_id = p['pair_id']
    deg_audio, deg_sr = sf.read(p['degraded_path'], dtype='float32')
    if deg_audio.ndim > 1:
        deg_audio = deg_audio.mean(axis=1)
    if deg_sr != SR:
        deg_audio = librosa.resample(deg_audio, orig_sr=deg_sr, target_sr=SR)
    if len(deg_audio) > CLIP_SAMPLES:
        deg_audio = deg_audio[:CLIP_SAMPLES]
    elif len(deg_audio) < CLIP_SAMPLES:
        deg_audio = np.pad(deg_audio, (0, CLIP_SAMPLES - len(deg_audio)))

    ref_path = p.get('reference_path') or p.get('pristine_path')
    if ref_path and Path(ref_path).exists():
        ref_audio, ref_sr = sf.read(ref_path, dtype='float32')
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)
        if ref_sr != SR:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=SR)
        if len(ref_audio) > CLIP_SAMPLES:
            ref_audio = ref_audio[:CLIP_SAMPLES]
        elif len(ref_audio) < CLIP_SAMPLES:
            ref_audio = np.pad(ref_audio, (0, CLIP_SAMPLES - len(ref_audio)))
    else:
        ref_audio = deg_audio.copy()

    pt_path = METRICS_DATA / 'pairs' / f'{pair_id}.pt'
    if pt_path.exists():
        tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
        m_degraded = tensor[0].numpy()
        m_reference = tensor[1].numpy()
    else:
        m_degraded = extract_metrics_67d(deg_audio)
        m_reference = extract_metrics_67d(ref_audio)

    cluster_id = int(p.get('cluster_id_reference', p.get('cluster_id_degraded', 0)))

    pair_data.append({
        'pair_id': pair_id,
        'degraded_audio': deg_audio,
        'reference_audio': ref_audio,
        'm_degraded': m_degraded,
        'm_reference': m_reference,
        'cluster_id': cluster_id,
        'deg_params': all_deg_params[pair_id],
    })
    print(f"  {pair_id}: MSE={np.mean((m_degraded - m_reference)**2):.2f}, cluster={cluster_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Compute inverse degradation targets for all pairs
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  INVERSE DEGRADATION: computing targets for {len(pair_data)} pairs")
print(f"{'='*60}")

supervised_data = []  # (observation_143d, target_action_227d, mse)

for pi, pd in enumerate(pair_data):
    # Build observation
    oh = np.zeros(N_CLUSTERS_ONEHOT, dtype=np.float32)
    if 0 <= pd['cluster_id'] < N_CLUSTERS:
        oh[pd['cluster_id']] = 1.0
    else:
        oh[N_CLUSTERS] = 1.0
    obs = np.concatenate([pd['m_degraded'], pd['m_reference'], oh]).astype(np.float32)

    # Compute inverse action from degradation params
    inv_action = compute_inverse_action(pd['deg_params'])

    # Verify: apply inverse action and measure resulting MSE
    try:
        plugin_dicts = decode_action(inv_action)
        processed = apply_plugins(pd['degraded_audio'], SR, plugin_dicts)
        m_result = extract_metrics_67d(processed)
        mse = float(np.mean((m_result - pd['m_reference']) ** 2))
    except Exception as e:
        print(f"  [WARN] Pair {pi} ({pd['pair_id']}): inverse failed — {e}")
        mse = float('inf')

    supervised_data.append((obs, inv_action, mse))
    if pi < 10 or (pi + 1) % 50 == 0:
        print(f"  Pair {pi:>5}/{len(pair_data)}: inverse_mse={mse:.2f}, "
              f"action_norm={np.linalg.norm(inv_action):.3f}")

# Summary
print(f"\n{'='*60}")
print(f"  INVERSE DEGRADATION SUMMARY")
print(f"{'='*60}")
mses = np.array([mse for _, _, mse in supervised_data])
print(f"  Total pairs:     {len(supervised_data)}")
print(f"  MSE range:       [{mses.min():.2f}, {mses.max():.2f}]")
print(f"  MSE mean:        {mses.mean():.2f} ± {mses.std():.2f}")
print(f"  MSE median:      {np.median(mses):.2f}")
n_finite = np.sum(np.isfinite(mses))
print(f"  Valid pairs:     {n_finite}/{len(supervised_data)}")
print(f"{'='*60}")

# Save for next cell
import pickle
save_path = OUTPUT / 'supervised_targets.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(supervised_data, f)
print(f"\n  Saved targets to {save_path}")
