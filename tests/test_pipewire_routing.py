"""
tests/test_pipewire_routing.py
Sends a known sine tone through the PipeWire dry-bypass loopback and
mathematically verifies hardware-level passthrough integrity.

Gracefully skips when PipeWire or sounddevice is unavailable (CI / headless).
"""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip conditions — resolved once at import time
# ---------------------------------------------------------------------------

_PIPEWIRE_AVAILABLE = shutil.which("pw-cli") is not None

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    _SOUNDDEVICE_AVAILABLE = False

_NEEDS_LIVE_PIPEWIRE = not (_PIPEWIRE_AVAILABLE and _SOUNDDEVICE_AVAILABLE)
_LIVE_SKIP_REASON = (
    "PipeWire (pw-cli) is not installed" if not _PIPEWIRE_AVAILABLE
    else "sounddevice is not available (no audio backend)" if not _SOUNDDEVICE_AVAILABLE
    else ""
)

# Mark used on integration tests only — sanity tests always run
_live_pipewire = pytest.mark.skipif(
    _NEEDS_LIVE_PIPEWIRE,
    reason=_LIVE_SKIP_REASON or "Requires live PipeWire + sounddevice",
)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TONE_FREQ_HZ = 440
TONE_DURATION_SEC = 0.5
SAMPLE_RATE = 48000
CHANNELS = 1
CORRELATION_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sine(freq: float, duration: float, sr: int) -> np.ndarray:
    """Generate a mono sine-wave tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Peak of the normalised cross-correlation between two 1-D signals.
    Returns a value in [0, 1] where 1.0 = perfect match.
    """
    a = a - np.mean(a)
    b = b - np.mean(b)

    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm < 1e-12:
        return 0.0

    # Use FFT-based cross-correlation for speed
    n = len(a) + len(b) - 1
    fft_size = 1
    while fft_size < n:
        fft_size <<= 1

    corr = np.fft.irfft(np.fft.rfft(a, fft_size) * np.conj(np.fft.rfft(b, fft_size)))
    return float(np.max(np.abs(corr)) / norm)


def _pipewire_loopback_active() -> bool:
    """Check if the Faurge PipeWire loopback node is registered."""
    import subprocess
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "faurge-loopback" in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipeWireRouting:
    """Suite to verify audio passes through the Faurge loopback intact."""

    def test_sine_generation_sanity(self):
        """Sanity: generated tone has the expected shape and range."""
        tone = generate_sine(TONE_FREQ_HZ, TONE_DURATION_SEC, SAMPLE_RATE)
        expected_samples = int(SAMPLE_RATE * TONE_DURATION_SEC)

        assert tone.shape == (expected_samples,)
        assert tone.dtype == np.float32
        assert np.max(np.abs(tone)) <= 1.0 + 1e-6

    def test_cross_correlation_identical_signals(self):
        """Sanity: identical signals yield correlation ~1.0."""
        tone = generate_sine(TONE_FREQ_HZ, TONE_DURATION_SEC, SAMPLE_RATE)
        corr = normalized_cross_correlation(tone, tone)
        assert corr >= 0.999, f"Self-correlation should be ~1.0, got {corr}"

    def test_cross_correlation_orthogonal_signals(self):
        """Sanity: uncorrelated signals yield low correlation."""
        tone_a = generate_sine(TONE_FREQ_HZ, TONE_DURATION_SEC, SAMPLE_RATE)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(tone_a.shape).astype(np.float32)
        corr = normalized_cross_correlation(tone_a, noise)
        assert corr < 0.3, f"Noise correlation should be low, got {corr}"

    @_live_pipewire
    def test_loopback_passthrough(self):
        """
        Full integration: play a sine through the loopback and verify
        the captured signal matches via cross-correlation.

        This test requires:
        1. PipeWire running
        2. The Faurge loopback created via scripts/setup_pipewire.sh
        3. A working sounddevice backend
        """
        if not _pipewire_loopback_active():
            pytest.skip("Faurge PipeWire loopback is not active — run setup_pipewire.sh first")

        tone = generate_sine(TONE_FREQ_HZ, TONE_DURATION_SEC, SAMPLE_RATE)
        record_duration = TONE_DURATION_SEC + 0.2  # slight padding
        record_samples = int(SAMPLE_RATE * record_duration)

        # Record while playing
        captured = sd.playrec(
            tone.reshape(-1, 1),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        captured_mono = captured.flatten()[:len(tone)]

        # Verify the passthrough
        corr = normalized_cross_correlation(tone, captured_mono)
        assert corr >= CORRELATION_THRESHOLD, (
            f"Loopback passthrough correlation {corr:.4f} "
            f"is below threshold {CORRELATION_THRESHOLD}. "
            f"The dry-bypass path may be broken."
        )

    @_live_pipewire
    def test_loopback_silence_when_no_input(self):
        """
        With no signal being played, the loopback capture should be
        near-silent (RMS < -60 dBFS).
        """
        if not _pipewire_loopback_active():
            pytest.skip("Faurge PipeWire loopback is not active")

        silence_sec = 0.3
        silence_samples = int(SAMPLE_RATE * silence_sec)

        captured = sd.rec(
            silence_samples,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        rms = float(np.sqrt(np.mean(captured ** 2)))
        # -60 dBFS ≈ 0.001
        assert rms < 0.001, (
            f"Expected near-silence but measured RMS={rms:.6f}. "
            f"There may be feedback or noise in the loopback."
        )
