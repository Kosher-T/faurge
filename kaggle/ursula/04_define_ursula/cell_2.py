# %% [markdown]
# ## Parameter Specification & Utilities
#
# Defines all 188 DSP parameters (EQ 186D + Gain 2D), their ranges, and the ActionUnnormalizer
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

# Gain: 2D
GAIN_PARAM_RANGES = [
    ParamRange("gain_db",           -12.0,     12.0),
    ParamRange("stereo_balance",     -1.0,      1.0),
]

# Master list: all 188D in order
ALL_PARAM_RANGES: List[ParamRange] = (
    EQ_PARAM_RANGES + GAIN_PARAM_RANGES
)
assert len(ALL_PARAM_RANGES) == OUTPUT_DIM

# ══════════════════════════════════════════════════════════════════════════════
# Plugin Slices & Head Dims
# ══════════════════════════════════════════════════════════════════════════════

PLUGIN_SLICES: Dict[str, Tuple[int, int]] = {}
PLUGIN_HEAD_DIMS: Dict[str, int] = {}
_offset = 0
for _name, _count in [
    ("eq", 31 * 6), ("gain", 2),
]:
    PLUGIN_SLICES[_name] = (_offset, _offset + _count)
    PLUGIN_HEAD_DIMS[_name] = _count
    _offset += _count

PLUGIN_HEAD_ORDER: List[str] = [
    "eq", "gain",
]

# ══════════════════════════════════════════════════════════════════════════════
# Categorical Parameter Indices
# ══════════════════════════════════════════════════════════════════════════════

CATEGORICAL_INDICES: Dict[str, List[int]] = {
    "eq_filter_type": list(range(2, 186, 6)),
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
