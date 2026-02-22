"""
core/hardware_detect.py
Probes for NVIDIA/AMD GPUs at startup.
If no supported GPU is found, forces Faurge into CPU-only mode,
disabling VRAM monitoring and switching all tensor inference to CPU/OpenVINO.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.logging import get_logger

log = get_logger("faurge.hardware_detect")


# ==============================================================================
# --- Data Classes ---
# ==============================================================================

@dataclass
class GPUInfo:
    """Holds detected GPU information."""
    vendor: str                     # "nvidia", "amd"
    name: str                       # e.g. "NVIDIA GeForce GTX 1650"
    vram_total_mb: int              # Total VRAM in MB
    driver_version: Optional[str] = None
    pci_bus_id: Optional[str] = None


@dataclass
class HardwareProfile:
    """
    The immutable hardware profile generated at startup.
    All downstream modules read this to decide their execution path.
    """
    gpu_available: bool = False
    cpu_only_mode: bool = True
    gpu: Optional[GPUInfo] = None
    system_ram_mb: int = 0
    cpu_count: int = 0
    os_name: str = ""
    kernel: str = ""
    warnings: list[str] = field(default_factory=list)


# ==============================================================================
# --- Detection Functions ---
# ==============================================================================

def _detect_nvidia() -> Optional[GPUInfo]:
    """Attempt to detect an NVIDIA GPU using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return None

        # Use the first GPU (index 0) — multi-GPU is out of scope for 4GB target
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total_mb = int(mem_info.total / (1024 * 1024))

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")

        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = pci_info.busId
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode("utf-8")

        pynvml.nvmlShutdown()

        return GPUInfo(
            vendor="nvidia",
            name=name,
            vram_total_mb=vram_total_mb,
            driver_version=driver_version,
            pci_bus_id=pci_bus_id,
        )
    except ImportError:
        log.debug("pynvml not installed — skipping NVIDIA detection.")
        return None
    except Exception as e:
        log.debug("NVIDIA detection failed: %s", e)
        return None


def _detect_amd() -> Optional[GPUInfo]:
    """
    Attempt to detect an AMD GPU via Linux sysfs.
    Reads /sys/class/drm/card*/device/vendor and mem_info_vram_total.
    """
    drm_base = Path("/sys/class/drm")
    if not drm_base.exists():
        return None

    AMD_VENDOR_ID = "0x1002"

    for card_dir in sorted(drm_base.iterdir()):
        vendor_file = card_dir / "device" / "vendor"
        if not vendor_file.exists():
            continue

        try:
            vendor_id = vendor_file.read_text().strip()
            if vendor_id != AMD_VENDOR_ID:
                continue

            # Found an AMD device — read VRAM
            vram_file = card_dir / "device" / "mem_info_vram_total"
            vram_total_mb = 0
            if vram_file.exists():
                vram_bytes = int(vram_file.read_text().strip())
                vram_total_mb = int(vram_bytes / (1024 * 1024))

            # Read device name from uevent
            name = "AMD GPU"
            uevent_file = card_dir / "device" / "uevent"
            if uevent_file.exists():
                for line in uevent_file.read_text().splitlines():
                    if line.startswith("PCI_SLOT_NAME="):
                        pci_bus_id = line.split("=", 1)[1]
                        name = f"AMD GPU ({pci_bus_id})"
                        break

            return GPUInfo(
                vendor="amd",
                name=name,
                vram_total_mb=vram_total_mb,
            )
        except (OSError, ValueError) as e:
            log.debug("AMD sysfs read error for %s: %s", card_dir.name, e)
            continue

    return None


def _get_system_ram_mb() -> int:
    """Returns total system RAM in MB."""
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(mem_bytes / (1024 * 1024))
    except (ValueError, OSError):
        return 0


# ==============================================================================
# --- Public API ---
# ==============================================================================

def detect_hardware() -> HardwareProfile:
    """
    Runs the full hardware probe and returns an immutable HardwareProfile.
    This is the single entry point called at Faurge startup.
    """
    profile = HardwareProfile(
        system_ram_mb=_get_system_ram_mb(),
        cpu_count=os.cpu_count() or 1,
        os_name=platform.system(),
        kernel=platform.release(),
    )

    if platform.system() != "Linux":
        profile.warnings.append(
            f"Unsupported OS: {platform.system()}. "
            "Faurge is currently Linux-only."
        )
        log.warning("Unsupported OS detected: %s", platform.system())
        return profile

    # Try NVIDIA first, then AMD
    gpu = _detect_nvidia()
    if gpu is None:
        gpu = _detect_amd()

    if gpu is not None:
        profile.gpu_available = True
        profile.cpu_only_mode = False
        profile.gpu = gpu
        log.info(
            "GPU detected: %s (%s) — %d MB VRAM",
            gpu.name, gpu.vendor.upper(), gpu.vram_total_mb,
        )
    else:
        profile.cpu_only_mode = True
        profile.warnings.append(
            "No supported GPU found. Running in CPU-only mode. "
            "VRAM monitoring disabled; inference will use CPU/OpenVINO."
        )
        log.warning("No GPU detected. Entering CPU-only mode.")

    return profile


# Module-level singleton so other modules can import it directly.
# Re-run detect_hardware() explicitly if hardware changes at runtime.
HARDWARE = detect_hardware()
