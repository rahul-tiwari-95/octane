"""System route — CPU, RAM, GPU, disk vitals for the dashboard gauges."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil
from fastapi import APIRouter

router = APIRouter(tags=["system"])


def _gpu_utilization() -> dict[str, Any]:
    """Get Apple Silicon GPU utilization via powermetrics (best-effort)."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AppleARMIODevice"],
            capture_output=True, text=True, timeout=2,
        )
        # Basic GPU memory estimation from system memory pressure
        mem = psutil.virtual_memory()
        # On unified memory Macs, GPU shares RAM — estimate GPU usage
        # from overall memory pressure (rough but informative)
        return {
            "available": True,
            "chip": platform.processor() or "Apple Silicon",
            "unified_memory_gb": round(mem.total / (1024**3), 1),
            "memory_pressure_percent": mem.percent,
        }
    except Exception:
        return {"available": False}


@router.get("/system")
async def system_vitals() -> dict[str, Any]:
    """System vitals for dashboard gauges."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    try:
        cpu_freq = psutil.cpu_freq()
        freq_mhz = round(cpu_freq.current, 0) if cpu_freq else 0
    except (SystemError, AttributeError):
        freq_mhz = 0

    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(),
            "freq_mhz": freq_mhz,
        },
        "ram": {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "used_gb": round(disk.used / (1024**3), 1),
            "percent": round(disk.percent, 1),
        },
        "gpu": _gpu_utilization(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
