# %% [markdown]
# ## Parameter Specification & Utilities
#
# Defines all 227 DSP parameters, their ranges, and the ActionUnnormalizer
# that converts tanh output to real plugin values.

# ══════════════════════════════════════════════════════════════════════════════
# Parameter Range Specification
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParamRange:
    """Single parameter: name, low/high bounds, log-scale flag."""
    name: str
    low: float
    high: float
    log: bool = False

# EQ: 31 bands × 6 params = 186D
EQ_PARAM_RANGES: List[ParamRange] = []
for _b in range(31):
    EQ_PARAM_RANGES.extend([
        ParamRange(f"eq_band{_b+1}_freq",        20.0,   20_000.0, log=True),
        ParamRange(f"eq_band{_b+1}_gain",       -24.0,      24.0),
        ParamRange(f"eq_band{_b+1}_q",            0.1,       10.0),
        ParamRange(f"eq_band{_b+1}_filter_type",  0.0,        6.0),
        ParamRange(f"eq_band{_b+1}_stereo_skew", -6.0,        6.0),
        ParamRange(f"eq_band{_b+1}_dynamic_depth", 0.0,        1.0),
    ])

# Compressor: 14D
COMP_PARAM_RANGES = [
    ParamRange("comp_threshold",      -60.0,     0.0),
    ParamRange("comp_ratio",           1.0,     20.0),
    ParamRange("comp_attack",          0.1,    100.0),
    ParamRange("comp_release",        10.0,   1000.0),
    ParamRange("comp_knee",            0.0,     12.0),
    ParamRange("comp_lookahead",       0.0,     10.0),
    ParamRange("comp_hold",            0.0,    200.0),
    ParamRange("comp_wet_dry",         0.0,      1.0),
    ParamRange("comp_stereo_link",     0.0,      1.0),
    ParamRange("comp_sidechain_hp",   20.0,    500.0),
    ParamRange("comp_sidechain_lp",  500.0,  20_000.0, log=True),
    ParamRange("comp_saturate_drive",  0.0,     12.0),
    ParamRange("comp_output_trim",   -12.0,     12.0),
    ParamRange("comp_detector_type",   0.0,      3.0),
]

# Esser: 6D
ESSER_PARAM_RANGES = [
    ParamRange("esser_center",       4000.0,  10_000.0, log=True),
    ParamRange("esser_threshold",    -60.0,      0.0),
    ParamRange("esser_ratio",          0.25,    20.0),
    ParamRange("esser_bandwidth",    500.0,   4000.0, log=True),
    ParamRange("esser_attack",         0.1,     50.0),
    ParamRange("esser_release",       10.0,    500.0),
]

# Saturator: 7D
SAT_PARAM_RANGES = [
    ParamRange("sat_drive",           0.0,     24.0),
    ParamRange("sat_mix",             0.0,      1.0),
    ParamRange("sat_type",            0.0,      3.0),
    ParamRange("sat_hpf",            20.0,    500.0),
    ParamRange("sat_lpf",          2000.0,  20_000.0, log=True),
    ParamRange("sat_oversampling",    0.0,      3.0),
    ParamRange("sat_output_trim",   -12.0,     12.0),
]

# Limiter: 6D
LIM_PARAM_RANGES = [
    ParamRange("lim_ceiling",       -12.0,      0.0),
    ParamRange("lim_release",         1.0,    500.0),
    ParamRange("lim_lookahead",       0.0,     10.0),
    ParamRange("lim_clip_mode",       0.0,      1.0),
    ParamRange("lim_stereo_link",     0.0,      1.0),
    ParamRange("lim_oversampling",    0.0,      3.0),
]

# Transient: 6D
TRANS_PARAM_RANGES = [
    ParamRange("trans_attack_gain",  -24.0,     24.0),
    ParamRange("trans_sustain_gain", -24.0,     24.0),
    ParamRange("trans_attack_time",    0.1,     50.0),
    ParamRange("trans_release_time",  10.0,    500.0),
    ParamRange("trans_sensitivity",  -30.0,      0.0),
    ParamRange("trans_mix",            0.0,      1.0),
]

# Gain: 2D
GAIN_PARAM_RANGES = [
    ParamRange("gain_db",           -12.0,     12.0),
    ParamRange("stereo_balance",     -1.0,      1.0),
]

# Master list: all 227D in order
ALL_PARAM_RANGES: List[ParamRange] = (
    EQ_PARAM_RANGES + COMP_PARAM_RANGES + ESSER_PARAM_RANGES
    + SAT_PARAM_RANGES + LIM_PARAM_RANGES + TRANS_PARAM_RANGES
    + GAIN_PARAM_RANGES
)
assert len(ALL_PARAM_RANGES) == OUTPUT_DIM

# ══════════════════════════════════════════════════════════════════════════════
# Plugin Slices & Head Dims
# ══════════════════════════════════════════════════════════════════════════════

PLUGIN_SLICES: Dict[str, Tuple[int, int]] = {}
PLUGIN_HEAD_DIMS: Dict[str, int] = {}
_offset = 0
for _name, _count in [
    ("eq", 31 * 6), ("compressor", 14), ("esser", 6),
    ("saturator", 7), ("limiter", 6), ("transient", 6), ("gain", 2),
]:
    PLUGIN_SLICES[_name] = (_offset, _offset + _count)
    PLUGIN_HEAD_DIMS[_name] = _count
    _offset += _count

PLUGIN_HEAD_ORDER: List[str] = [
    "eq", "compressor", "esser", "saturator", "limiter", "transient", "gain",
]

# ══════════════════════════════════════════════════════════════════════════════
# Categorical Parameter Indices
# ══════════════════════════════════════════════════════════════════════════════

CATEGORICAL_INDICES: Dict[str, List[int]] = {
    "eq_filter_type": list(range(2, 186, 6)),
    "comp_detector_type": [186 + 13],
    "sat_type": [206 + 2],
    "sat_oversampling": [206 + 5],
    "lim_clip_mode": [213 + 3],
    "lim_oversampling": [213 + 5],
}

# ══════════════════════════════════════════════════════════════════════════════
# Pre-computed Tensors for Vectorized Decode
# ══════════════════════════════════════════════════════════════════════════════

_PARAM_LOWS = torch.tensor([pr.low for pr in ALL_PARAM_RANGES], dtype=torch.float32)
PARAM_HIGHS = torch.tensor([pr.high for pr in ALL_PARAM_RANGES], dtype=torch.float32)
_PARAM_IS_LOG = torch.tensor([pr.log for pr in ALL_PARAM_RANGES], dtype=torch.bool)
_CAT_MASK = torch.zeros(OUTPUT_DIM, dtype=torch.bool)
for _indices in CATEGORICAL_INDICES.values():
    _CAT_MASK[_indices] = True

print(f"Param ranges: {len(ALL_PARAM_RANGES)}")
print(f"Categorical groups: {len(CATEGORICAL_INDICES)}")
print(f"Plugin heads: {PLUGIN_HEAD_DIMS}")
