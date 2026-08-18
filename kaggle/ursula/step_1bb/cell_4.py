# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Degradation Function
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
import equalizer as eq_plugin

def freq_to_input(freq_hz):
    """Normalize frequency to 0-1 range (log scale)."""
    return (np.log(freq_hz) - np.log(FREQ_MIN_HZ)) / (np.log(FREQ_MAX_HZ) - np.log(FREQ_MIN_HZ))

def degrade_with_eq(audio, freq_hz, sr=SR):
    """Apply EQ at given frequency with random gain. Return (degraded_audio, gain_db)."""
    gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
    degraded, _ = eq_plugin.process(
        audio, sr=sr,
        bands=[{
            "freq_hz": float(freq_hz),
            "gain_db": float(gain_db),
            "q": EQ_FIXED_Q,
            "filter_type": "peak"
        }]
    )
    return degraded, np.float32(gain_db)

print(f"Degradation: random gain {EQ_GAIN_RANGE_DB}, fixed Q={EQ_FIXED_Q}")
print(f"Frequency normalization: log({FREQ_MIN_HZ})-log({FREQ_MAX_HZ}) → 0-1")
