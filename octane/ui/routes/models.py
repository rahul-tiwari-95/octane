"""Models route — loaded Bodega models, memory, context windows."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["models"])

_BODEGA_URL = "http://localhost:44468"


async def _fetch_loaded_models() -> list[dict[str, Any]]:
    """Fetch loaded models from Bodega admin endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_BODEGA_URL}/v1/admin/loaded-models")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("data", []):
                mem = m.get("memory", {})
                models.append({
                    "id": m["id"],
                    "type": m.get("type", "lm"),
                    "context_length": m.get("context_length", 4096),
                    "status": m.get("status", "unknown"),
                    "memory_mb": mem.get("total_mb", 0),
                    "pid": m.get("pid"),
                })
            return models
    except Exception:
        return []


async def _fetch_health() -> dict[str, Any]:
    """Fetch Bodega health status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_BODEGA_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "offline"}


@router.get("/models")
async def get_models() -> dict[str, Any]:
    """Currently loaded Bodega models."""
    models = await _fetch_loaded_models()
    health = await _fetch_health()
    return {
        "models": models,
        "engine_status": health.get("status", "offline"),
        "total_models": len(models),
    }
