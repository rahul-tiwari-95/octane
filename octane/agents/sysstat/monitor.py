"""Monitor — system resource metrics via psutil."""

from __future__ import annotations

import psutil


class Monitor:
    """Collects system resource metrics.

    Uses psutil for RAM, CPU. Token metrics will be added when
    Bodega queue stats are integrated.
    """

    def snapshot(self) -> dict:
        """Take a snapshot of current system resources.

        Returns:
            Dict with ram_total_gb, ram_used_gb, ram_available_gb,
            ram_percent, cpu_percent, cpu_count.
        """
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)

        return {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_available_gb": round(mem.available / (1024**3), 2),
            "ram_percent": mem.percent,
            "cpu_percent": cpu_pct,
            "cpu_count": psutil.cpu_count(),
        }
