"""Model Manager — loads/unloads models via Bodega admin API.

ONLY this component calls Bodega admin endpoints. No other agent
manages models directly.
"""

from __future__ import annotations

from typing import Any

import structlog

from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.topology import Topology

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

    async def reload_with_reasoning_parser(
        self,
        reasoning_parser: str = "qwen3",
        context_length: int = 32768,
    ) -> dict[str, Any]:
        """Unload the current model and reload it with reasoning_parser enabled.

        This is the one-time setup needed to get native reasoning_content in
        API responses instead of raw <think>...</think> tokens in content.

        Args:
            reasoning_parser: Parser name — "qwen3" for bodega-raptor-8b-mxfp4
            context_length: Context window size. Default 32768. Lower = less RAM.

        Returns:
            Load result dict from Bodega.

        Raises:
            RuntimeError if no model is currently loaded.
        """
        info = await self.current_model()
        if not info.get("loaded"):
            raise RuntimeError("reload_with_reasoning_parser: no model currently loaded")

        model_path = info["model_info"]["model_path"]
        current_ctx = info["model_info"].get("context_length", context_length)

        logger.info(
            "reloading_with_reasoning_parser",
            model=model_path,
            parser=reasoning_parser,
            context_length=current_ctx,
        )

        await self.unload_model()
        result = await self.load_model(
            model_path,
            model_type="lm",
            context_length=current_ctx,
            reasoning_parser=reasoning_parser,
        )
        logger.info("model_reloaded_with_parser", model=model_path, result=result)
        return result

    async def ensure_topology_loaded(
        self, topology: Topology
    ) -> dict[str, dict[str, Any]]:
        """Load every model tier defined in *topology* via the Bodega admin API.

        Uses the full :class:`~octane.tools.topology.ModelConfig` for each tier
        — including ``max_concurrency``, ``prompt_cache_size``, ``context_length``,
        and speculative-decoding params (``draft_model_path`` / ``num_draft_tokens``).

        Deduplicates by ``model_id``: if FAST and MID share the same model (as in
        ``compact`` and ``balanced``), the model is loaded only once.

        Args:
            topology: A fully-configured :class:`~octane.tools.topology.Topology`
                      instance (e.g. from :func:`~octane.tools.topology.get_topology`).

        Returns:
            A dict mapping tier name → ``{"status": "loaded"|"skipped"|"failed", ...}``.
        """
        results: dict[str, dict[str, Any]] = {}
        seen_model_ids: set[str] = set()

        for tier, config in topology.models.items():
            tier_name = tier.value

            if config.model_id in seen_model_ids:
                logger.debug(
                    "ensure_topology_skip_duplicate",
                    tier=tier_name,
                    model_id=config.model_id,
                )
                results[tier_name] = {
                    "status": "skipped",
                    "reason": "duplicate_model_id",
                    "model_id": config.model_id,
                }
                continue

            params = config.to_load_params()
            logger.info(
                "ensure_topology_loading",
                tier=tier_name,
                model_id=config.model_id,
                max_concurrency=config.max_concurrency,
                prompt_cache_size=config.prompt_cache_size,
                speculative=config.draft_model_path is not None,
            )
            try:
                await self.load_model(**params)
                seen_model_ids.add(config.model_id)
                results[tier_name] = {
                    "status": "loaded",
                    "model_id": config.model_id,
                    "max_concurrency": config.max_concurrency,
                    "prompt_cache_size": config.prompt_cache_size,
                    "speculative_decoding": config.draft_model_path is not None,
                }
                logger.info(
                    "ensure_topology_loaded",
                    tier=tier_name,
                    model_id=config.model_id,
                )
            except Exception as exc:
                logger.warning(
                    "ensure_topology_load_failed",
                    tier=tier_name,
                    model_id=config.model_id,
                    error=str(exc),
                )
                results[tier_name] = {
                    "status": "failed",
                    "model_id": config.model_id,
                    "error": str(exc),
                }

        return results
