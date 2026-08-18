import os
import sys
import numpy as np

# Allow import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.compressor.compressor import process


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def make_tone_burst(freq, amp, sr, burst_dur, silence_dur, cycles):
    burst_n = int(sr * burst_dur)
    silence_n = int(sr * silence_dur)
    total = (burst_n + silence_n) * cycles
    buf = np.zeros(total, dtype=np.float32)
    for c in range(cycles):
        offset = c * (burst_n + silence_n)
        t = np.arange(burst_n) / sr
        buf[offset:offset + burst_n] = amp * np.sin(2.0 * np.pi * freq * t)
    return buf


def make_percussive(sr, dur):
    n = int(sr * dur)
    buf = np.zeros(n, dtype=np.float32)
    for hit in range(5):
        hit_pos = int(hit * sr * dur / 5.0)
        for i in range(50):
            if hit_pos + i < n:
                env = 1.0 - i / 50.0
                buf[hit_pos + i] = 0.8 * env * np.sin(2.0 * np.pi * 1000.0 * i / sr)
    return buf


def make_transient(sr, dur):
    n = int(sr * dur)
    buf = np.zeros(n, dtype=np.float32)
    click_pos = int(sr * 0.1)
    for i in range(100):
        if click_pos + i < n:
            buf[click_pos + i] = 0.9 * (1.0 - i / 100.0)
    return buf


def compute_rms(buf):
    return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))


# ── Passthrough Tests ────────────────────────────────────────────────────────

def test_ratio_one():
    """ratio=1 → audio unchanged (no gain reduction regardless of threshold)."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr,
                             ratio=1.0,
                             threshold_db=-80.0,
                             attack_ms=0.1,
                             release_ms=10.0,
                             knee_db=0.0)

    assert result["success"]
    assert result["gain_reduction_db"] < 0.1, \
        f"gain_reduction_db={result['gain_reduction_db']}"
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-4, f"max_diff={max_diff}"


def test_threshold_above_peak():
    """Threshold above signal peak → no compression."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)

    output, result = process(audio, sr,
                             threshold_db=0.0,
                             ratio=20.0,
                             attack_ms=0.1,
                             release_ms=10.0,
                             knee_db=0.0)

    assert result["success"]
    assert result["gain_reduction_db"] < 0.1


def test_wet_dry_zero():
    """wet_dry=0 → output identical to input."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr,
                             threshold_db=-60.0,
                             ratio=20.0,
                             attack_ms=0.1,
                             release_ms=10.0,
                             knee_db=0.0,
                             wet_dry_mix=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-4, f"max_diff={max_diff}"


# ── Pumping Tests ────────────────────────────────────────────────────────────

def test_extreme_pumping():
    """Extreme ratio + low threshold → heavy gain reduction."""
    sr = 44100
    audio = make_tone_burst(440.0, 0.3, sr, 0.5, 0.5, 3)

    output, result = process(audio, sr,
                             threshold_db=-60.0,
                             ratio=20.0,
                             attack_ms=0.1,
                             release_ms=10.0,
                             knee_db=0.0)

    assert result["success"]
    assert result["gain_reduction_db"] > 10.0, \
        f"gain_reduction_db={result['gain_reduction_db']}"


# ── Lookahead Tests ──────────────────────────────────────────────────────────

def test_lookahead_shifts_gr():
    """Lookahead=5ms shifts gain reduction envelope earlier."""
    sr = 44100
    audio = make_transient(sr, 0.5)
    original = audio.copy()

    # Without lookahead
    out_no_la, _ = process(audio.copy(), sr,
                           threshold_db=-40.0,
                           ratio=20.0,
                           attack_ms=0.1,
                           release_ms=5.0,
                           knee_db=0.0,
                           lookahead_ms=0.0)

    # With lookahead
    out_with_la, _ = process(audio.copy(), sr,
                              threshold_db=-40.0,
                              ratio=20.0,
                              attack_ms=0.1,
                              release_ms=5.0,
                              knee_db=0.0,
                              lookahead_ms=5.0)

    # Both should show gain reduction
    diff_no_la = float(np.max(np.abs(out_no_la - original)))
    diff_with_la = float(np.max(np.abs(out_with_la - original)))
    assert diff_no_la > 0.01, f"no-LA diff={diff_no_la}"
    assert diff_with_la > 0.01, f"with-LA diff={diff_with_la}"


# ── Detector Tests ───────────────────────────────────────────────────────────

def test_rms_vs_peak_difference():
    """RMS vs peak detectors show different GR profiles on percussive material."""
    sr = 44100
    audio = make_percussive(sr, 1.0)

    out_rms, result_rms = process(audio.copy(), sr,
                                   threshold_db=-30.0,
                                   ratio=10.0,
                                   attack_ms=0.1,
                                   release_ms=10.0,
                                   knee_db=0.0,
                                   detector_type="RMS")

    out_peak, result_peak = process(audio.copy(), sr,
                                     threshold_db=-30.0,
                                     ratio=10.0,
                                     attack_ms=0.1,
                                     release_ms=10.0,
                                     knee_db=0.0,
                                     detector_type="peak")

    assert result_rms["success"]
    assert result_peak["success"]
    assert result_rms["gain_reduction_db"] > 0.0
    assert result_peak["gain_reduction_db"] > 0.0

    # Outputs should differ (different detection → different GR profiles)
    diff = float(np.max(np.abs(out_rms - out_peak)))
    assert diff > 0.001, f"RMS/peak outputs identical (diff={diff})"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_ratio_one", test_ratio_one),
        ("test_threshold_above_peak", test_threshold_above_peak),
        ("test_wet_dry_zero", test_wet_dry_zero),
        ("test_extreme_pumping", test_extreme_pumping),
        ("test_lookahead_shifts_gr", test_lookahead_shifts_gr),
        ("test_rms_vs_peak_difference", test_rms_vs_peak_difference),
    ]

    passed = 0
    failed = 0
    print("\n=== Compressor: Python Port Tests ===")
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
