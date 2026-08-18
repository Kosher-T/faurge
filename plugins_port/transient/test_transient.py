import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.transient.transient import process


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def make_percussive(sr, dur):
    n = int(sr * dur)
    buf = np.zeros(n, dtype=np.float32)
    for hit in range(5):
        hit_pos = int(hit * sr * dur / 5.0)
        for i in range(100):
            if hit_pos + i < n:
                env = 1.0 - i / 100.0
                buf[hit_pos + i] = 0.8 * env * np.sin(2.0 * np.pi * 1000.0 * i / sr)
    return buf


def compute_rms(buf):
    return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))


def compute_peak(buf):
    return float(np.max(np.abs(buf)))


def compute_peak_to_rms(buf):
    rms = compute_rms(buf)
    if rms < 1e-30:
        return 0.0
    return 20.0 * np.log10(compute_peak(buf) / rms)


# ── Gain=0, Mix=0 → unchanged ───────────────────────────────────────────────

def test_gain_mix_zero_unchanged():
    """attack_gain=0, sustain_gain=0, mix=0 → audio unchanged."""
    sr = 44100
    audio = make_percussive(sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr,
                             attack_gain_db=0.0,
                             sustain_gain_db=0.0,
                             mix=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-4, f"max_diff={max_diff}"


# ── Attack boost increases peak-to-RMS ───────────────────────────────────────

def test_attack_boost_increases_peak_to_rms():
    """Positive attack_gain on percussive audio increases peak-to-RMS ratio."""
    sr = 44100
    audio = make_percussive(sr, 1.0)

    out_no_boost, _ = process(audio.copy(), sr,
                              attack_gain_db=0.0,
                              sustain_gain_db=0.0,
                              mix=1.0)
    out_boosted, _ = process(audio.copy(), sr,
                             attack_gain_db=12.0,
                             sustain_gain_db=0.0,
                             mix=1.0)

    p2r_no = compute_peak_to_rms(out_no_boost)
    p2r_boosted = compute_peak_to_rms(out_boosted)
    assert p2r_boosted > p2r_no, \
        f"Expected boosted peak-to-RMS ({p2r_boosted:.1f} dB) > no boost ({p2r_no:.1f} dB)"


def test_sustain_increases_peak_to_rms():
    """Negative sustain_gain increases peak-to-RMS ratio (relatively more transient)."""
    sr = 44100
    audio = make_percussive(sr, 1.0)

    out_no_change, _ = process(audio.copy(), sr,
                               attack_gain_db=0.0,
                               sustain_gain_db=0.0,
                               mix=1.0)
    out_attack_boost, _ = process(audio.copy(), sr,
                                  attack_gain_db=0.0,
                                  sustain_gain_db=-12.0,
                                  mix=1.0)

    p2r_no = compute_peak_to_rms(out_no_change)
    p2r_reduced = compute_peak_to_rms(out_attack_boost)
    assert p2r_reduced > p2r_no, \
        f"Expected sustain reduction to increase peak-to-RMS: {p2r_reduced:.1f} vs {p2r_no:.1f} dB"


# ── Silence stability ────────────────────────────────────────────────────────

def test_silence_in_silence_out():
    """Silence in → silence out (no runaway oscillation)."""
    sr = 44100
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)

    output, result = process(audio, sr,
                             attack_gain_db=24.0,
                             sustain_gain_db=24.0,
                             attack_time_ms=1.0,
                             release_time_ms=10.0,
                             mix=1.0)

    assert result["success"]
    max_val = float(np.max(np.abs(output)))
    assert max_val < 1e-6, f"Silence output should be near zero, got {max_val}"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_gain_mix_zero_unchanged", test_gain_mix_zero_unchanged),
        ("test_attack_boost_increases_peak_to_rms", test_attack_boost_increases_peak_to_rms),
        ("test_sustain_increases_peak_to_rms", test_sustain_increases_peak_to_rms),
        ("test_silence_in_silence_out", test_silence_in_silence_out),
    ]

    passed = 0
    failed = 0
    print("\n=== Transient: Python Port Tests ===")
    for name, fn in tests:
        print(f"  [RUN]  {name}")
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    sys.exit(1 if failed > 0 else 0)
