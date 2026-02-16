"""SysStat Agent coordinator.

Reports system metrics, manages model loading, and adaptive scaling.
This is the ONLY agent that calls Bodega admin endpoints.
"""

from __future__ import annotations

from octane.agents.base import BaseAgent
from octane.agents.sysstat.monitor import Monitor
from octane.agents.sysstat.model_manager import ModelManager
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.bodega_inference import BodegaInferenceClient


class SysStatAgent(BaseAgent):
    """SysStat Agent — system health and model management.

    Sub-agents:
        - Monitor: RAM/CPU/token metrics via psutil
        - ModelManager: loads/unloads models via Bodega admin API
        - Scaler: adaptive model topology (stub)
    """

    name = "sysstat"

    def __init__(self, synapse, bodega: BodegaInferenceClient | None = None) -> None:
        super().__init__(synapse)
        self.bodega = bodega or BodegaInferenceClient()
        self.monitor = Monitor()
        self.model_manager = ModelManager(self.bodega)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Gather system metrics and model info."""
        # System metrics from psutil
        system_metrics = self.monitor.snapshot()

        # Model info from Bodega
        model_info = await self.model_manager.current_model()
        server_health = await self.bodega.health()

        return AgentResponse(
            agent=self.name,
            success=True,
            output=self._format_output(system_metrics, model_info, server_health),
            data={
                "system": system_metrics,
                "model": model_info,
                "server_health": server_health,
            },
        )

    def _format_output(
        self,
        system: dict,
        model: dict,
        health: dict,
    ) -> str:
        """Format system status as human-readable text."""
        lines = [
            "── System Status ──",
            f"  RAM: {system.get('ram_used_gb', '?'):.1f} / {system.get('ram_total_gb', '?'):.1f} GB "
            f"({system.get('ram_percent', '?')}%)",
            f"  CPU: {system.get('cpu_percent', '?')}%",
            "",
            "── Bodega Inference Engine ──",
            f"  Server: {health.get('status', 'unknown')}",
        ]

        if "error" in model:
            lines.append(f"  Model: ⚠ {model['error']}")
        else:
            model_name = model.get("model", model.get("model_path", "unknown"))
            lines.append(f"  Model: {model_name}")

        return "\n".join(lines)
