"""Catalysts — pre-written deterministic code blocks for common tasks.

Catalysts bypass the LLM code-generation pipeline entirely. For a recognised
query pattern they receive structured upstream data, run pure Python, and
return an AgentResponse directly. No <think> blocks, no invented filenames,
no sandbox timeout risk.

Usage (inside CodeAgent):
    from octane.catalysts.registry import CatalystRegistry
    registry = CatalystRegistry()
    result = registry.match(query, upstream_results)
    if result:
        catalyst_fn, resolved_data = result
        return catalyst_fn(resolved_data, output_dir, correlation_id)
"""
from octane.catalysts.registry import CatalystRegistry

__all__ = ["CatalystRegistry"]
