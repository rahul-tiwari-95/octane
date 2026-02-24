"""Scaler — adaptive model topology recommendations based on available RAM.

Reads available RAM from Monitor, maps to the best model topology for the hardware.
Does NOT auto-load models in Session 9 — recommendations only.
Model hot-swap is Phase 3.

Topology tiers (based on available RAM headroom):
    tier_64gb : 40GB+ available  → 30B brain + 8B worker + 0.9B grunt
    tier_32gb : 20–40GB available → 8B brain + 4B worker + 0.9B grunt
    tier_16gb : 8–20GB available  → 8B brain + 0.9B grunt
    tier_8gb  : < 8GB available   → 0.9B brain only
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="sysstat.scaler")

# Model identifiers used by Bodega Inference Engine
MODEL_TOPOLOGIES: dict[str, dict] = {
    "tier_64gb": {
        "description": "High memory — full 3-model stack",
        "min_ram_gb": 40,
        "brain":  "SRSWTI/axe-turbo-31b",
        "worker": "SRSWTI/bodega-raptor-8b-mxfp4",
        "grunt":  "SRSWTI/bodega-raptor-0.9b",
    },
    "tier_32gb": {
        "description": "Mid memory — 8B brain + light worker",
        "min_ram_gb": 20,
        "brain":  "SRSWTI/bodega-raptor-8b-mxfp4",
        "worker": "SRSWTI/bodega-vertex-4b",
        "grunt":  "SRSWTI/bodega-raptor-0.9b",
    },
    "tier_16gb": {
        "description": "Low memory — 8B brain + grunt only",
        "min_ram_gb": 8,
        "brain": "SRSWTI/bodega-raptor-8b-mxfp4",
        "grunt": "SRSWTI/bodega-raptor-0.9b",
    },
    "tier_8gb": {
        "description": "Minimal memory — single small model",
        "min_ram_gb": 0,
        "brain": "SRSWTI/bodega-raptor-0.9b",
    },
}

# Ordered from highest to lowest for tier selection
_TIER_ORDER = ["tier_64gb", "tier_32gb", "tier_16gb", "tier_8gb"]


class Scaler:
    """Recommends a model topology based on available RAM.

    Reads Monitor.snapshot()['ram_available_gb'] and returns the highest
    tier whose min_ram_gb requirement is met.
    """

    def recommend(self, ram_available_gb: float) -> dict:
        """Select the best model topology for the given available RAM.

        Returns a dict with:
            tier        — tier name (e.g. "tier_32gb")
            description — human-readable label
            models      — {role: model_id} for each loaded model
            ram_gb      — the available RAM value used for selection
        """
        selected = MODEL_TOPOLOGIES["tier_8gb"]
        selected_name = "tier_8gb"

        for tier_name in _TIER_ORDER:
            tier = MODEL_TOPOLOGIES[tier_name]
            if ram_available_gb >= tier["min_ram_gb"]:
                selected = tier
                selected_name = tier_name
                break

        models = {
            role: model_id
            for role, model_id in selected.items()
            if role not in ("description", "min_ram_gb")
        }

        logger.info(
            "scaler_recommendation",
            tier=selected_name,
            ram_available_gb=round(ram_available_gb, 1),
        )

        return {
            "tier": selected_name,
            "description": selected["description"],
            "models": models,
            "ram_gb": round(ram_available_gb, 1),
        }

    # Kept for backward compatibility with any callers using the old async signature
    async def recommend_topology(self, ram_available_gb: float) -> dict:
        return self.recommend(ram_available_gb)
