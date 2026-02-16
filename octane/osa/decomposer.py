"""OSA Decomposer — query → TaskDAG.

Phase 1: Simple heuristic-based decomposition.
Phase 2+: Uses big model to reason about query structure.
"""

from __future__ import annotations

import structlog

from octane.models.dag import TaskDAG, TaskNode

logger = structlog.get_logger().bind(component="osa.decomposer")


class Decomposer:
    """Decomposes a user query into a TaskDAG.

    Phase 1: Simple keyword-based routing to a single agent.
    Phase 2+: LLM-powered multi-step DAG generation.
    """

    async def decompose(self, query: str) -> TaskDAG:
        """Analyze the query and produce a task DAG.

        Phase 1 logic:
            - Keywords like 'stock', 'price', 'news' → Web Agent
            - Keywords like 'write', 'code', 'script' → Code Agent
            - Keywords like 'remember', 'recall', 'memory' → Memory Agent
            - Keywords like 'health', 'status', 'system' → SysStat Agent
            - Default → Web Agent (search)
        """
        query_lower = query.lower()

        agent, reasoning = self._classify_query(query_lower)

        dag = TaskDAG(
            original_query=query,
            reasoning=reasoning,
            nodes=[
                TaskNode(
                    agent=agent,
                    instruction=query,
                    metadata={"source": "decomposer_v1"},
                ),
            ],
        )

        logger.info(
            "decomposed",
            query=query[:100],
            agent=agent,
            reasoning=reasoning,
        )
        return dag

    def _classify_query(self, query_lower: str) -> tuple[str, str]:
        """Simple keyword-based query classification.

        Returns:
            (agent_name, reasoning)
        """
        # SysStat triggers
        sysstat_keywords = ["health", "status", "system", "ram", "cpu", "model", "loaded"]
        if any(kw in query_lower for kw in sysstat_keywords):
            return "sysstat", "Query is about system health or resource status"

        # Code triggers
        code_keywords = ["write", "code", "script", "program", "function", "implement",
                         "fibonacci", "algorithm", "python", "javascript"]
        if any(kw in query_lower for kw in code_keywords):
            return "code", "Query asks for code generation or programming"

        # Memory triggers
        memory_keywords = ["remember", "recall", "memory", "forget", "stored", "saved"]
        if any(kw in query_lower for kw in memory_keywords):
            return "memory", "Query is about stored memory or recall"

        # Finance triggers
        finance_keywords = ["stock", "price", "market", "ticker", "portfolio", "shares",
                           "nasdaq", "nyse", "earnings"]
        if any(kw in query_lower for kw in finance_keywords):
            return "web", "Query is about financial data — routing to Web Agent"

        # News triggers
        news_keywords = ["news", "headlines", "happening", "today", "latest"]
        if any(kw in query_lower for kw in news_keywords):
            return "web", "Query is about current events — routing to Web Agent"

        # Default: general search via Web Agent
        return "web", "General query — routing to Web Agent for search"
