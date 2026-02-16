"""Model Manager — loads/unloads models via Bodega admin API.

ONLY this component calls Bodega admin endpoints. No other agent
manages models directly.
"""

from __future__ import annotations

from typing import Any

import structlog

from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="model_manager")


class ModelManager:
    """Manages LLM model lifecycle via Bodega Inference Engine.

    Responsible for:
        - Checking which model is loaded
        - Loading/unloading models based on RAM and workload
        - Implementing model topology strategies
    """

    def __init__(self, bodega: BodegaInferenceClient) -> None:
        self.bodega = bodega

    async def current_model(self) -> dict[str, Any]:
        """Get info about the currently loaded model."""
        return await self.bodega.current_model()

    async def list_models(self) -> dict[str, Any]:
        """List all available/cached models."""
        return await self.bodega.list_models()

    async def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """Load a model into the inference engine."""
        logger.info("loading_model", model_path=model_path)
        result = await self.bodega.load_model(model_path, **kwargs)
        logger.info("model_loaded", result=result)
        return result

    async def unload_model(self) -> dict[str, Any]:
        """Unload the current model."""
        logger.info("unloading_model")
        result = await self.bodega.unload_model()
        logger.info("model_unloaded", result=result)
        return result
