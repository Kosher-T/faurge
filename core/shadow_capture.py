"""
core/shadow_capture.py
Asynchronous 5-second ringbuffer recorder.
Captures live audio from PipeWire into the Shadow Space for downstream
agent analysis. The buffer is continuously overwritten until a snapshot
is explicitly requested.
"""

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from core import defaults
from core.logging import get_logger

log = get_logger("faurge.shadow_capture")


# ==============================================================================
# --- Ring Buffer ---
# ==============================================================================

class AudioRingBuffer:
    """
    A fixed-size circular buffer for raw PCM audio samples.
    Thread-safe for one writer + one reader.
    """

    def __init__(self, max_samples: int, channels: int = 1):
        self.max_samples = max_samples
        self.channels = channels
        self._buffer = np.zeros((max_samples, channels), dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """
        Append audio samples to the ring buffer.
        Args:
            data: numpy array of shape (N,) or (N, channels).
        """
        if data.ndim == 1:
            data = data.reshape(-1, self.channels)

        n = data.shape[0]
        with self._lock:
            if n >= self.max_samples:
                # Data larger than buffer — keep only the tail
                self._buffer[:] = data[-self.max_samples:]
                self._write_pos = 0
            elif self._write_pos + n <= self.max_samples:
                self._buffer[self._write_pos : self._write_pos + n] = data
                self._write_pos += n
            else:
                # Wrap around
                first_chunk = self.max_samples - self._write_pos
                self._buffer[self._write_pos :] = data[:first_chunk]
                remaining = n - first_chunk
                self._buffer[:remaining] = data[first_chunk:]
                self._write_pos = remaining

            self._total_written += n

    def snapshot(self) -> np.ndarray:
        """
        Returns the current buffer contents in chronological order.
        If the buffer isn't full yet, returns only the written portion.
        """
        with self._lock:
            if self._total_written < self.max_samples:
                # Buffer not yet full
                return self._buffer[: self._write_pos].copy()
            else:
                # Buffer has wrapped — re-order so oldest samples come first
                return np.concatenate([
                    self._buffer[self._write_pos :],
                    self._buffer[: self._write_pos],
                ]).copy()

    @property
    def duration_samples(self) -> int:
        """Number of valid samples currently in the buffer."""
        return min(self._total_written, self.max_samples)


# ==============================================================================
# --- Shadow Capture Service ---
# ==============================================================================

class ShadowCapture:
    """
    Manages asynchronous audio capture into a ring buffer.
    Uses sounddevice (via PipeWire ALSA or native backend) to record.
    """

    def __init__(
        self,
        sample_rate: int = defaults.AUDIO_SAMPLE_RATE,
        channels: int = defaults.AUDIO_CHANNELS,
        buffer_length_sec: float = defaults.SHADOW_BUFFER_LENGTH_SEC,
        shadow_dir: Optional[Path] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_length_sec = buffer_length_sec

        max_samples = int(sample_rate * buffer_length_sec)
        self.ring_buffer = AudioRingBuffer(max_samples, channels)

        if shadow_dir is None:
            from core.settings import DIRS
            shadow_dir = DIRS["SHADOW"]
        self.shadow_dir = Path(shadow_dir)
        self.shadow_dir.mkdir(parents=True, exist_ok=True)

        self._stream = None
        self._running = False

        log.info(
            "ShadowCapture initialized: %d Hz, %d ch, %.1fs buffer (%d samples)",
            sample_rate, channels, buffer_length_sec, max_samples,
        )

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            log.warning("Audio callback status: %s", status)
        self.ring_buffer.write(indata.copy())

    def start(self) -> None:
        """Begin capturing audio from the default PipeWire input."""
        if self._running:
            log.warning("ShadowCapture is already running.")
            return

        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            log.info("Shadow capture started — recording to ring buffer.")
        except Exception as e:
            log.error("Failed to start shadow capture: %s", e)
            self._running = False
            raise

    def stop(self) -> None:
        """Stop the audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
                self._running = False
                log.info("Shadow capture stopped.")

    def take_snapshot(self, filename: Optional[str] = None) -> Path:
        """
        Freeze the current ring buffer contents and write to a WAV file
        in the Shadow Space directory.

        Args:
            filename: Optional custom filename. Defaults to timestamped name.

        Returns:
            Path to the written WAV file.
        """
        audio_data = self.ring_buffer.snapshot()

        if audio_data.shape[0] == 0:
            log.warning("Ring buffer is empty — no snapshot taken.")
            raise RuntimeError("Cannot take snapshot: ring buffer is empty.")

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"shadow_{timestamp}.wav"

        output_path = self.shadow_dir / filename

        sf.write(str(output_path), audio_data, self.sample_rate)
        duration_sec = audio_data.shape[0] / self.sample_rate
        log.info(
            "Shadow snapshot saved: %s (%.2fs, %d samples)",
            output_path.name, duration_sec, audio_data.shape[0],
        )

        return output_path

    @property
    def is_running(self) -> bool:
        return self._running
