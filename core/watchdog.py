"""
core/watchdog.py
The VRAM monitoring daemon for Faurge.
Runs as a persistent loop (designed for systemd), polling GPU memory usage
and emitting structured log events. Gracefully degrades on GPU-less systems.
"""

import signal
import sys
import time
from typing import Optional

from core import defaults
from core.logging import get_logger
from core.hardware_detect import HARDWARE

log = get_logger("faurge.watchdog")


# ==============================================================================
# --- VRAM Poller ---
# ==============================================================================

class VRAMPoller:
    """
    Periodically polls GPU VRAM usage and emits structured metrics.
    In CPU-only mode, monitors system RAM instead.
    """

    def __init__(
        self,
        poll_rate_sec: float = defaults.WATCHDOG_POLL_RATE_SEC,
        vram_safety_margin_mb: int = defaults.VRAM_SAFETY_MARGIN_MB,
    ):
        self.poll_rate_sec = poll_rate_sec
        self.vram_safety_margin_mb = vram_safety_margin_mb
        self._running = False
        self._nvml_handle = None
        self._cpu_only = HARDWARE.cpu_only_mode

        if not self._cpu_only:
            self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize pynvml for GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            log.info("VRAM Watchdog initialized for GPU: %s", HARDWARE.gpu.name)
        except Exception as e:
            log.warning(
                "Failed to initialize pynvml — falling back to CPU-only monitoring: %s", e
            )
            self._cpu_only = True
            self._nvml_handle = None

    def poll_once(self) -> dict:
        """
        Performs a single poll and returns a metrics dict.
        Returns:
            dict with keys: vram_used_mb, vram_total_mb, vram_free_mb,
                            utilization_pct, budget_exceeded (bool)
        """
        if self._cpu_only:
            return self._poll_ram()
        return self._poll_vram()

    def _poll_vram(self) -> dict:
        """Poll NVIDIA VRAM via pynvml."""
        try:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            used_mb = int(mem_info.used / (1024 * 1024))
            total_mb = int(mem_info.total / (1024 * 1024))
            free_mb = total_mb - used_mb
            utilization_pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
            budget_limit_mb = total_mb - self.vram_safety_margin_mb

            return {
                "mode": "gpu",
                "vram_used_mb": used_mb,
                "vram_total_mb": total_mb,
                "vram_free_mb": free_mb,
                "utilization_pct": utilization_pct,
                "budget_limit_mb": budget_limit_mb,
                "budget_exceeded": used_mb > budget_limit_mb,
            }
        except Exception as e:
            log.error("VRAM poll failed: %s", e)
            return {"mode": "gpu", "error": str(e)}

    def _poll_ram(self) -> dict:
        """Fallback: poll system RAM from /proc/meminfo."""
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        val_kb = int(parts[1])
                        meminfo[key] = val_kb

            total_mb = meminfo.get("MemTotal", 0) // 1024
            available_mb = meminfo.get("MemAvailable", 0) // 1024
            used_mb = total_mb - available_mb
            utilization_pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0

            return {
                "mode": "cpu_ram",
                "ram_used_mb": used_mb,
                "ram_total_mb": total_mb,
                "ram_available_mb": available_mb,
                "utilization_pct": utilization_pct,
                "budget_exceeded": False,  # RAM budget not enforced until Phase 2
            }
        except Exception as e:
            log.error("RAM poll failed: %s", e)
            return {"mode": "cpu_ram", "error": str(e)}

    def start(self) -> None:
        """
        Starts the polling loop. Blocks the calling thread.
        Intended to be run as a daemon (systemd service).
        """
        self._running = True
        mode_label = "GPU VRAM" if not self._cpu_only else "System RAM (CPU-only)"
        log.info("Watchdog started — monitoring %s every %.1fs", mode_label, self.poll_rate_sec)

        while self._running:
            metrics = self.poll_once()
            if "error" not in metrics:
                if metrics.get("budget_exceeded"):
                    log.warning("⚠ VRAM budget exceeded! %s", metrics)
                else:
                    log.debug("Watchdog poll: %s", metrics)
            time.sleep(self.poll_rate_sec)

        log.info("Watchdog stopped.")

    def stop(self) -> None:
        """Signal the polling loop to exit gracefully."""
        self._running = False

    def shutdown(self) -> None:
        """Clean up pynvml resources."""
        self.stop()
        if not self._cpu_only and self._nvml_handle is not None:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                log.info("pynvml shutdown complete.")
            except Exception:
                pass


# ==============================================================================
# --- Daemon Entry Point ---
# ==============================================================================

def main() -> None:
    """
    Entry point for the systemd service.
    Usage: python -m core.watchdog
    """
    poller = VRAMPoller()

    def _signal_handler(signum, frame):
        log.info("Received signal %d — shutting down watchdog.", signum)
        poller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    poller.start()


if __name__ == "__main__":
    main()
