"""OSA Decomposer — query → TaskDAG.

Session 2: LLM-powered intent classification with template selection.
Session 5: Single-token classification — LLM outputs exactly one template name,
           no JSON parsing needed. Reliable, fast, zero ambiguity.
Falls back to keyword heuristics if Bodega is unavailable.
"""

from __future__ import annotations

import structlog

from octane.models.dag import TaskDAG, TaskNode
from octane.osa.dag_planner import DAGPlanner

logger = structlog.get_logger().bind(component="osa.decomposer")

# ── Pipeline Templates ────────────────────────────────────────────────────────
# The LLM chooses from these. Adding a new capability = adding a template here.
# Format: template_name → (agent, sub_agent_hint, description)

PIPELINE_TEMPLATES: dict[str, dict] = {
    "web_finance": {
        "agent": "web",
        "sub_agent": "finance",
        "description": "Fetch stock price, market data, or financial metrics",
        "keywords": ["stock", "price", "market", "ticker", "shares", "nasdaq", "nyse", "earnings", "portfolio"],
    },
    "web_news": {
        "agent": "web",
        "sub_agent": "news",
        "description": "Search for recent news, headlines, or current events",
        "keywords": ["news", "headlines", "happening", "today", "latest", "announcement", "reported"],
    },
    "web_search": {
        "agent": "web",
        "sub_agent": "search",
        "description": "General web search or research question",
        "keywords": ["search", "find", "look up", "what is", "who is", "where is", "how does"],
    },
    "code_generation": {
        "agent": "code",
        "sub_agent": "full_pipeline",
        "description": "Write, generate, or implement code or a script",
        "keywords": ["write", "code", "script", "program", "function", "implement",
                     "fibonacci", "algorithm", "python", "javascript", "build"],
    },
    "memory_recall": {
        "agent": "memory",
        "sub_agent": "read",
        "description": "Recall something previously stored or discussed",
        "keywords": ["remember", "recall", "memory", "forget", "stored", "saved", "previously"],
    },
    "sysstat_health": {
        "agent": "sysstat",
        "sub_agent": "monitor",
        "description": "Check system health, RAM, CPU, or loaded model status",
        "keywords": ["health", "status", "system", "ram", "cpu", "model", "loaded", "memory usage"],
    },
}

_VALID_TEMPLATES = set(PIPELINE_TEMPLATES.keys())

# Single-token prompt: LLM must output exactly one template name — nothing else.
# No JSON, no reasoning, no markdown. Just the token. Reliable at temperature=0.
_DECOMPOSER_SYSTEM = (
    "You are a query router. Respond with ONLY one of these exact words, nothing else:\n"
    "web_finance, web_news, web_search, code_generation, memory_recall, sysstat_health\n\n"
    "Meanings:\n"
    "  web_finance    → stock price, market data, financial metrics, earnings\n"
    "  web_news       → recent news, headlines, current events, announcements\n"
    "  web_search     → general research, who/what/where questions, information lookup\n"
    "  code_generation → write code, script, program, algorithm, implementation\n"
    "  memory_recall  → recall something previously discussed or stored\n"
    "  sysstat_health → system health, CPU, RAM, model status\n\n"
    "If unsure, output: web_search"
)


class Decomposer:
    """Decomposes a user query into a TaskDAG.

    Session 2: Uses a small LLM to classify intent and select a pipeline
    template. Falls back to keyword heuristics if Bodega is unavailable.

    Phase 3+: Generates arbitrary multi-step parallel DAGs from scratch.
    """

    def __init__(self, bodega=None) -> None:
        # Bodega client injected by Orchestrator (avoids circular imports)
        self._bodega = bodega
        self._dag_planner = DAGPlanner(bodega=bodega)

    async def decompose(self, query: str) -> TaskDAG:
        """Analyze the query and produce a TaskDAG.

        For compound queries (fetch + code, multi-company, etc.) the DAGPlanner
        is tried first. Falls back to single-node classification if planning
        produces a trivial or invalid result.
        """
        # Attempt multi-node planning for compound queries
        if self._dag_planner.is_compound(query):
            dag = await self._dag_planner.plan(query)
            if dag is not None:
                logger.info(
                    "decomposed_multi",
                    query=query[:80],
                    nodes=len(dag.nodes),
                    waves=len(dag.execution_order()),
                )
                return dag

        # Single-node fallback
        template_name, reasoning, source = await self._classify(query)
        template = PIPELINE_TEMPLATES[template_name]

        dag = TaskDAG(
            original_query=query,
            reasoning=reasoning,
            nodes=[
                TaskNode(
                    agent=template["agent"],
                    instruction=query,
                    metadata={
                        "source": source,
                        "template": template_name,
                        "sub_agent": template["sub_agent"],
                    },
                ),
            ],
        )

        logger.info(
            "decomposed",
            query=query[:100],
            template=template_name,
            agent=template["agent"],
            source=source,
            reasoning=reasoning,
        )
        return dag

    async def _classify(self, query: str) -> tuple[str, str, str]:
        """Returns (template_name, reasoning, source).

        source is 'llm' or 'keyword_fallback'.
        """
        if self._bodega is not None:
            try:
                result = await self._classify_with_llm(query)
                if result:
                    return result[0], result[1], "llm"
            except Exception as exc:
                logger.warning("llm_classification_failed", error=str(exc), fallback="keywords")

        # Fallback: keyword heuristics
        template_name, reasoning = self._classify_with_keywords(query.lower())
        return template_name, reasoning, "keyword_fallback"

    async def _classify_with_llm(self, query: str) -> tuple[str, str] | None:
        """Ask the LLM to output a single template name token.

        No JSON parsing — just strip whitespace and check set membership.
        Handles reasoning models that emit <think>...</think> blocks.
        """
        raw = await self._bodega.chat_simple(
            prompt=f'Query: "{query}"',
            system=_DECOMPOSER_SYSTEM,
            temperature=0.0,   # deterministic — classification, not generation
            max_tokens=256,    # enough room for a thinking block + one token
        )

        # Strip <think>...</think> blocks emitted by reasoning models (DeepSeek-R1 etc.)
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = raw  # fallback: use raw if stripping removed everything

        template_name = cleaned.strip().lower().strip('"\'.,;:\n ')

        if template_name not in _VALID_TEMPLATES:
            logger.warning("llm_returned_unknown_template", raw=raw[:80])
            return None

        return template_name, f"LLM selected: {template_name}"

    def _classify_with_keywords(self, query_lower: str) -> tuple[str, str]:
        """Keyword-based fallback classification.

        Scores each template by how many of its keywords appear in the query.
        Returns the highest-scoring template (ties broken by order).
        """
        scores: dict[str, int] = {}
        for name, template in PIPELINE_TEMPLATES.items():
            score = sum(1 for kw in template["keywords"] if kw in query_lower)
            scores[name] = score

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "web_search", "No keyword match — defaulting to web search"

        return best, f"Keyword match for template '{best}' (score: {scores[best]})"
