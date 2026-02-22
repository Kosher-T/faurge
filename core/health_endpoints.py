"""
core/health_endpoints.py
Lightweight HTTP health/readiness/metrics server for Faurge.
Exposes /health, /ready, and /metrics endpoints using Python's built-in
http.server — no external framework required.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

from core import defaults
from core.logging import get_logger

log = get_logger("faurge.health")


# ==============================================================================
# --- Response Helpers ---
# ==============================================================================

def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    """Send a JSON response with the given HTTP status code."""
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


# ==============================================================================
# --- Health Request Handler ---
# ==============================================================================

class HealthHandler(BaseHTTPRequestHandler):
    """
    Routes:
        GET /health  — Liveness probe  (is the process alive?)
        GET /ready   — Readiness probe (are subsystems OK?)
        GET /metrics — JSON metrics blob for monitoring
    """

    # Reference to the parent HealthServer for shared state
    server: "HealthServer"

    # Suppress default stderr logging per request
    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    # --- Route dispatch ---

    def do_GET(self):
        path = self.path.rstrip("/")

        routes = {
            "/health": self._handle_health,
            "/ready": self._handle_ready,
            "/metrics": self._handle_metrics,
        }

        handler_fn = routes.get(path)
        if handler_fn:
            handler_fn()
        else:
            _json_response(self, 404, {"error": "not_found", "path": self.path})

    # --- Endpoint implementations ---

    def _handle_health(self):
        """Liveness: always 200 if the process is running."""
        _json_response(self, 200, {
            "status": "ok",
            "timestamp": time.time(),
        })

    def _handle_ready(self):
        """
        Readiness: checks that downstream subsystems are usable.
        - Hardware detection completed
        - Watchdog is importable
        """
        issues: list[str] = []

        # Check hardware detection
        try:
            from core.hardware_detect import HARDWARE
            hw_status = {
                "gpu_available": HARDWARE.gpu_available,
                "cpu_only_mode": HARDWARE.cpu_only_mode,
                "system_ram_mb": HARDWARE.system_ram_mb,
            }
            if HARDWARE.gpu_available and HARDWARE.gpu:
                hw_status["gpu_name"] = HARDWARE.gpu.name
        except Exception as exc:
            issues.append(f"hardware_detect: {exc}")
            hw_status = {"error": str(exc)}

        # Check that PipeWire tools are reachable (basic sanity)
        import shutil
        pw_available = shutil.which("pw-cli") is not None
        if not pw_available:
            issues.append("pw-cli not found on PATH")

        ready = len(issues) == 0
        status_code = 200 if ready else 503

        _json_response(self, status_code, {
            "ready": ready,
            "hardware": hw_status,
            "pipewire_cli": pw_available,
            "issues": issues,
            "timestamp": time.time(),
        })

    def _handle_metrics(self):
        """
        Returns a JSON blob with watchdog metrics, hardware profile,
        and uptime — suitable for Prometheus-style scraping.
        """
        metrics: dict[str, Any] = {
            "timestamp": time.time(),
            "uptime_sec": round(time.time() - self.server.start_time, 2),
        }

        # Hardware profile
        try:
            from core.hardware_detect import HARDWARE
            metrics["hardware"] = {
                "gpu_available": HARDWARE.gpu_available,
                "cpu_only_mode": HARDWARE.cpu_only_mode,
                "cpu_count": HARDWARE.cpu_count,
                "system_ram_mb": HARDWARE.system_ram_mb,
                "os": HARDWARE.os_name,
                "kernel": HARDWARE.kernel,
            }
            if HARDWARE.gpu_available and HARDWARE.gpu:
                metrics["hardware"]["gpu"] = {
                    "vendor": HARDWARE.gpu.vendor,
                    "name": HARDWARE.gpu.name,
                    "vram_total_mb": HARDWARE.gpu.vram_total_mb,
                    "driver_version": HARDWARE.gpu.driver_version,
                }
        except Exception as exc:
            metrics["hardware"] = {"error": str(exc)}

        # Watchdog snapshot (single poll)
        try:
            from core.watchdog import VRAMPoller
            poller = VRAMPoller()
            metrics["watchdog"] = poller.poll_once()
        except Exception as exc:
            metrics["watchdog"] = {"error": str(exc)}

        _json_response(self, 200, metrics)


# ==============================================================================
# --- Health Server ---
# ==============================================================================

class HealthServer:
    """
    Manages the HTTP health server lifecycle.
    Runs in a daemon thread so it doesn't block the main Faurge loop.
    """

    def __init__(self, port: Optional[int] = None):
        self.port = port or defaults.HEALTH_HTTP_PORT
        self.start_time = time.time()
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the health server in a background daemon thread."""
        if self._httpd is not None:
            log.warning("Health server is already running on port %d.", self.port)
            return

        self._httpd = HTTPServer(("0.0.0.0", self.port), HealthHandler)
        # Store start_time on httpd so handler instances can access it
        self._httpd.start_time = self.start_time

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="faurge-health",
            daemon=True,
        )
        self._thread.start()
        log.info("Health server started on http://0.0.0.0:%d", self.port)

    def stop(self) -> None:
        """Shut down the health server gracefully."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            self._thread = None
            log.info("Health server stopped.")

    @property
    def is_running(self) -> bool:
        return self._httpd is not None


# ==============================================================================
# --- Standalone entry point ---
# ==============================================================================

def main() -> None:
    """Run the health server standalone for testing."""
    import signal
    import sys

    server = HealthServer()
    server.start()

    print(f"Health server listening on http://0.0.0.0:{server.port}")
    print("Endpoints: /health  /ready  /metrics")
    print("Press Ctrl+C to stop.\n")

    def _shutdown(signum, frame):
        print("\nShutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block the main thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
