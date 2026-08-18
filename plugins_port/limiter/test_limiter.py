import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.limiter.limiter import process


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def make_impulse(amp, sr, dur):
    n = int(sr * dur)
    buf = np.zeros(n, dtype=np.float32)
    buf[0] = amp
    return buf


def compute_rms(buf):
    return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))


def compute_peak(buf):
    return float(np.max(np.abs(buf)))


# ── Soft Limit ───────────────────────────────────────────────────────────────

def test_soft_limit_near_invisible():
    """High ceiling + soft → near-invisible limiting."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr,
                             ceiling_db=-0.1,
                             release_ms=100.0,
                             lookahead_ms=0.0,
                             clip_mode="soft")

    assert result["success"]
    assert result["max_gain_reduction_db"] < 3.0, \
        f"Expected minimal GR, got {result['max_gain_reduction_db']}"
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 0.1, f"Output changed too much: max_diff={max_diff}"


# ── Hard Limit ───────────────────────────────────────────────────────────────

def test_hard_limit_measurable_gr():
    """Low ceiling + hard → measurable gain reduction."""
    sr = 44100
    audio = make_sine(440.0, 0.9, sr, 1.0)

    output, result = process(audio, sr,
                             ceiling_db=-6.0,
                             release_ms=10.0,
                             lookahead_ms=0.0,
                             clip_mode="hard")

    assert result["success"]
    assert result["max_gain_reduction_db"] > 3.0, \
        f"Expected significant GR, got {result['max_gain_reduction_db']}"
    output_peak = compute_peak(output)
    ceiling_lin = 10.0 ** (-6.0 / 20.0)
    assert output_peak <= ceiling_lin + 0.01, \
        f"Output peak {output_peak:.4f} exceeds ceiling {ceiling_lin:.4f}"


# ── Ceiling Compliance ───────────────────────────────────────────────────────

def test_ceiling_compliance():
    """Output peak never exceeds ceiling_db (100 random trials)."""
    sr = 44100
    ceiling_db = -3.0
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    np.random.seed(42)

    for trial in range(100):
        amp = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(100, 8000)
        audio = make_sine(freq, amp, sr, 0.1)

        output, result = process(audio, sr,
                                 ceiling_db=ceiling_db,
                                 release_ms=np.random.uniform(10, 200),
                                 lookahead_ms=np.random.uniform(0, 10),
                                 clip_mode="hard")

        assert result["success"]
        peak = compute_peak(output)
        assert peak <= ceiling_lin + 0.001, \
            f"Trial {trial}: peak {peak:.6f} exceeds ceiling {ceiling_lin:.6f}"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_soft_limit_near_invisible", test_soft_limit_near_invisible),
        ("test_hard_limit_measurable_gr", test_hard_limit_measurable_gr),
        ("test_ceiling_compliance", test_ceiling_compliance),
    ]

    passed = 0
    failed = 0
    print("\n=== Limiter: Python Port Tests ===")
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
