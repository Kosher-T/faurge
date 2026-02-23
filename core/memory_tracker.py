"""
core/memory_tracker.py
Detailed peak and per-layer memory profiling for Faurge Phase 2.

Supports both VRAM (pynvml) and system RAM (psutil) tracking.
Used by scripts/vram_budget.py to measure real memory consumption
during model loading and inference, and to debug budget overruns.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from core.hardware_detect import HARDWARE
from core.logging import get_logger

log = get_logger("faurge.memory_tracker")


# ==============================================================================
# --- Data Classes ---
# ==============================================================================

@dataclass
class MemorySnapshot:
    """Point-in-time memory reading."""
    timestamp: float            # time.monotonic() value
    used_mb: float              # Current usage in MB
    mode: str                   # 'gpu' or 'cpu_ram'
    label: Optional[str] = None # Optional human-readable label


@dataclass
class LayerDelta:
    """Memory delta attributed to a named block."""
    name: str
    before_mb: float
    after_mb: float
    delta_mb: float


# ==============================================================================
# --- Memory Tracker ---
# ==============================================================================

class MemoryTracker:
    """
    Tracks memory usage across operations, recording snapshots and
    per-layer deltas.

    Usage:
        tracker = MemoryTracker()
        tracker.snapshot("baseline")

        with tracker.track_layer("model_load"):
            model = load_model(...)

        with tracker.track_layer("inference"):
            model.predict(...)

        print(tracker.report())
    """

    def __init__(self, mode: Optional[str] = None):
        """
        Args:
            mode: 'gpu' or 'cpu_ram'. Auto-detected from HardwareProfile
                  if not specified.
        """
        if mode is not None:
            self._mode = mode
        else:
            self._mode = "cpu_ram" if HARDWARE.cpu_only_mode else "gpu"

        self._snapshots: list[MemorySnapshot] = []
        self._layer_deltas: list[LayerDelta] = []
        self._peak_mb: float = 0.0
        self._nvml_handle = None

        if self._mode == "gpu":
            self._init_nvml()

        log.info("MemoryTracker initialized — mode=%s", self._mode)

    # ------------------------------------------------------------------
    # GPU backend
    # ------------------------------------------------------------------

    def _init_nvml(self) -> None:
        """Initialize pynvml for VRAM tracking."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            log.warning(
                "pynvml init failed — falling back to CPU-only tracking: %s", e
            )
            self._mode = "cpu_ram"
            self._nvml_handle = None

    def _read_gpu_mb(self) -> float:
        """Read current GPU VRAM usage in MB."""
        try:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return mem_info.used / (1024 * 1024)
        except Exception as e:
            log.error("GPU memory read failed: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # CPU / RAM backend
    # ------------------------------------------------------------------

    @staticmethod
    def _read_ram_mb() -> float:
        """Read current system RAM usage in MB via psutil."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.used / (1024 * 1024)
        except Exception as e:
            log.error("RAM read failed: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # Unified read
    # ------------------------------------------------------------------

    def _read_mb(self) -> float:
        """Dispatch to the correct backend."""
        if self._mode == "gpu":
            return self._read_gpu_mb()
        return self._read_ram_mb()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self, label: Optional[str] = None) -> MemorySnapshot:
        """
        Take a point-in-time memory reading and store it.

        Args:
            label: Optional tag for this snapshot (e.g. 'baseline', 'post_load')

        Returns:
            The MemorySnapshot that was recorded.
        """
        used = self._read_mb()
        snap = MemorySnapshot(
            timestamp=time.monotonic(),
            used_mb=round(used, 2),
            mode=self._mode,
            label=label,
        )
        self._snapshots.append(snap)

        if used > self._peak_mb:
            self._peak_mb = used

        log.debug("Snapshot [%s]: %.2f MB (%s)", label or "?", used, self._mode)
        return snap

    @contextmanager
    def track_layer(self, name: str):
        """
        Context manager that records the memory delta for a named block.

        Usage:
            with tracker.track_layer("encoder"):
                encoder = build_encoder(...)
        """
        before = self._read_mb()
        yield
        after = self._read_mb()

        delta = LayerDelta(
            name=name,
            before_mb=round(before, 2),
            after_mb=round(after, 2),
            delta_mb=round(after - before, 2),
        )
        self._layer_deltas.append(delta)

        if after > self._peak_mb:
            self._peak_mb = after

        log.debug(
            "Layer [%s]: %.2f → %.2f MB (Δ %+.2f MB)",
            name, before, after, delta.delta_mb,
        )

    @property
    def peak_mb(self) -> float:
        """Return the highest memory reading observed so far."""
        return round(self._peak_mb, 2)

    @property
    def mode(self) -> str:
        """Return the tracking mode ('gpu' or 'cpu_ram')."""
        return self._mode

    def report(self) -> dict:
        """
        Returns a structured summary of all tracked data.

        Keys:
            mode, peak_mb, baseline_mb, snapshots, layers
        """
        baseline = self._snapshots[0].used_mb if self._snapshots else 0.0

        return {
            "mode": self._mode,
            "peak_mb": self.peak_mb,
            "baseline_mb": baseline,
            "snapshot_count": len(self._snapshots),
            "snapshots": [
                {
                    "label": s.label,
                    "used_mb": s.used_mb,
                }
                for s in self._snapshots
            ],
            "layers": [
                {
                    "name": ld.name,
                    "before_mb": ld.before_mb,
                    "after_mb": ld.after_mb,
                    "delta_mb": ld.delta_mb,
                }
                for ld in self._layer_deltas
            ],
        }

    def shutdown(self) -> None:
        """Release pynvml resources if initialized."""
        if self._mode == "gpu" and self._nvml_handle is not None:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                log.info("MemoryTracker pynvml shutdown complete.")
            except Exception:
                pass
