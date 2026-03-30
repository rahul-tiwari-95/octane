"""Command Mapper — natural language → Octane CLI decomposition.

Session 37: Crown Jewel Chat.

Translates natural language into one or more Octane internal operations.
Uses a command manifest describing every capability, and an LLM to match
user intent to the right operation(s) with parameters.

The mapper does NOT shell out to CLI commands — it calls the underlying
Python functions directly for speed and structured data return.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import structlog

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="command_mapper")


@dataclass
class MappedCommand:
    """A single resolved Octane operation."""
    operation: str          # e.g. "portfolio.show", "research.start"
    description: str        # human-readable: "Show your portfolio positions with live prices"
    parameters: dict = field(default_factory=dict)  # e.g. {"ticker": "AAPL", "prices": True}


@dataclass
class CommandPlan:
    """The mapper's output: one or more commands + reasoning."""
    reasoning: list[str]    # bullet points showing bunny's thinking
    commands: list[MappedCommand]
    original_query: str


# ── Command Manifest ─────────────────────────────────────────────────────────
# Every Octane capability the mapper can invoke.
# Format: operation_id → (description, example_triggers, parameter_schema)

COMMAND_MANIFEST = {
    "portfolio.show": {
        "description": "Show portfolio positions with optional live prices",
        "triggers": [
            "show my portfolio", "show me my portfolio", "my positions", "my holdings",
            "what stocks do I own", "how is my portfolio", "portfolio performance",
            "my investments", "portfolio",
        ],
        "params": {"prices": "bool — fetch live prices (default true)"},
    },
    "portfolio.import": {
        "description": "Import portfolio positions from a CSV file",
        "triggers": [
            "import portfolio", "import my positions", "load my trades",
            "add positions from file",
        ],
        "params": {"file": "str — path to CSV file"},
    },
    "research.start": {
        "description": "Start a deep research cycle on a topic",
        "triggers": [
            "research", "look into", "dig into", "investigate",
            "find out about", "what can you find on",
        ],
        "params": {"topic": "str — research subject", "depth": "str — quick|normal|deep"},
    },
    "research.compare": {
        "description": "Compare two or more items side by side",
        "triggers": [
            "compare", "versus", "vs", "head to head",
            "which is better", "difference between",
        ],
        "params": {"items": "list[str] — items to compare", "criteria": "str — comparison focus"},
    },
    "web.search": {
        "description": "Search the web for current information",
        "triggers": [
            "search for", "look up", "find me", "google",
            "what is", "who is", "latest on",
        ],
        "params": {"query": "str — search query"},
    },
    "web.news": {
        "description": "Get latest news on a topic",
        "triggers": [
            "news about", "headlines", "what's happening with",
            "latest news", "current events",
        ],
        "params": {"query": "str — news topic"},
    },
    "web.finance": {
        "description": "Get stock price, market data, or financial info",
        "triggers": [
            "stock price", "how is X trading", "market data",
            "price of", "ticker", "what's X at",
        ],
        "params": {"ticker": "str — stock symbol or company name"},
    },
    "extract.url": {
        "description": "Extract and store content from a URL (article, PDF, video)",
        "triggers": [
            "extract", "read this article", "save this page",
            "download content from", "get the text from",
        ],
        "params": {"url": "str — URL to extract"},
    },
    "extract.arxiv": {
        "description": "Search arXiv for research papers",
        "triggers": [
            "arxiv", "research papers", "academic papers",
            "find papers on", "scientific papers about",
        ],
        "params": {"query": "str — search query", "max_results": "int — default 5"},
    },
    "recall.search": {
        "description": "Search through stored knowledge (web pages, extractions, research)",
        "triggers": [
            "search my knowledge", "what do I have on",
            "search my notes", "find in my saved",
        ],
        "params": {"query": "str — search terms"},
    },
    "recall.stats": {
        "description": "Show knowledge base statistics",
        "triggers": [
            "knowledge stats", "how much have I stored",
            "my knowledge base", "recall stats",
        ],
        "params": {},
    },
    "system.health": {
        "description": "Check system health: Bodega, Redis, Postgres, models",
        "triggers": [
            "system status", "health check", "is everything running",
            "check the system", "system health",
        ],
        "params": {},
    },
    "system.stats": {
        "description": "Show the Octane stats dashboard",
        "triggers": [
            "stats", "dashboard", "show me stats",
            "overview", "system overview",
        ],
        "params": {},
    },
    "watch.status": {
        "description": "Show status of background monitoring tasks",
        "triggers": [
            "monitoring status", "watch status", "what's being monitored",
            "background tasks", "active monitors",
        ],
        "params": {},
    },
    "model.list": {
        "description": "List loaded AI models",
        "triggers": [
            "what models", "loaded models", "which models",
            "available models", "model info",
        ],
        "params": {},
    },
    "pref.show": {
        "description": "Show user preferences",
        "triggers": [
            "my preferences", "my settings", "show preferences",
            "current settings",
        ],
        "params": {},
    },
    "pref.set": {
        "description": "Update a user preference",
        "triggers": [
            "set preference", "change setting", "update my",
        ],
        "params": {"key": "str — preference name", "value": "str — new value"},
    },
    "code.generate": {
        "description": "Generate code or a script",
        "triggers": [
            "write code", "write a script", "implement", "code for",
            "program that", "function to", "write me a",
        ],
        "params": {"description": "str — what to code", "language": "str — optional language"},
    },
    "vault.encrypt": {
        "description": "Encrypt a file with the Octane vault",
        "triggers": [
            "encrypt", "secure this file", "lock file", "vault encrypt",
        ],
        "params": {"file": "str — file path"},
    },
    "vault.decrypt": {
        "description": "Decrypt a vaulted file",
        "triggers": [
            "decrypt", "unlock file", "vault decrypt",
        ],
        "params": {"file": "str — file path"},
    },
}


# ── LLM system prompt for command mapping ────────────────────────────────────

def _build_manifest_prompt() -> str:
    lines = [
        "You are a command mapper for the Octane AI assistant.",
        "Given a user's natural language request, map it to one or more operations.",
        "",
        "Available operations:",
    ]
    for op_id, info in COMMAND_MANIFEST.items():
        params = ", ".join(f"{k}: {v}" for k, v in info.get("params", {}).items())
        params_str = f" | params: {params}" if params else ""
        lines.append(f"  {op_id} — {info['description']}{params_str}")

    lines.extend([
        "",
        "Respond with a JSON object:",
        '{',
        '  "reasoning": ["step 1...", "step 2..."],',
        '  "commands": [',
        '    {"operation": "op_id", "description": "what this does", "parameters": {...}}',
        '  ]',
        '}',
        "",
        "Rules:",
        "- Be aggressive: 'I am thinking about Apple' → research.start with topic 'Apple Inc'",
        "- Infer parameters from context: 'how is Tesla doing' → web.finance with ticker TSLA",
        "- Chain commands when useful: 'research and compare X and Y' → research.start + research.compare",
        "- Keep reasoning in simple English, max 3-4 bullet points",
        "- Output ONLY valid JSON, nothing else",
    ])
    return "\n".join(lines)


_MANIFEST_PROMPT = _build_manifest_prompt()


class CommandMapper:
    """Maps natural language to Octane operations."""

    def __init__(self, bodega=None) -> None:
        self._bodega = bodega

    async def map(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CommandPlan:
        """Map a natural language query to Octane commands.

        Falls back to keyword matching if LLM is unavailable.
        """
        # Try LLM mapping first
        if self._bodega is not None:
            try:
                plan = await asyncio.wait_for(
                    self._map_with_llm(query, conversation_history),
                    timeout=10.0,
                )
                if plan:
                    return plan
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("command_mapper_llm_failed", error=str(exc))

        # Fallback: keyword matching
        return self._map_with_keywords(query)

    async def _map_with_llm(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CommandPlan | None:
        """Use LLM to map query to commands."""
        context = ""
        if conversation_history:
            recent = conversation_history[-4:]
            parts = []
            for entry in recent:
                role = "User" if entry["role"] == "user" else "Assistant"
                parts.append(f"{role}: {entry['content'][:300]}")
            context = "Recent conversation:\n" + "\n".join(parts) + "\n\n"

        prompt = f'{context}User request: "{query}"'

        raw = await self._bodega.chat_simple(
            prompt=prompt,
            system=_MANIFEST_PROMPT,
            tier=ModelTier.REASON,
            temperature=0.1,
            max_tokens=512,
        )

        # Strip thinking tags
        if "</think>" in raw:
            _, _, raw = raw.partition("</think>")
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            logger.warning("command_mapper_no_json", raw=raw[:100])
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("command_mapper_bad_json", raw=raw[:100])
            return None

        reasoning = data.get("reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]

        commands = []
        for cmd_data in data.get("commands", []):
            op = cmd_data.get("operation", "")
            if op not in COMMAND_MANIFEST:
                logger.warning("command_mapper_unknown_op", op=op)
                continue
            commands.append(MappedCommand(
                operation=op,
                description=cmd_data.get("description", COMMAND_MANIFEST[op]["description"]),
                parameters=cmd_data.get("parameters", {}),
            ))

        if not commands:
            return None

        return CommandPlan(
            reasoning=reasoning,
            commands=commands,
            original_query=query,
        )

    def _map_with_keywords(self, query: str) -> CommandPlan:
        """Keyword-based fallback mapping."""
        q = query.lower()
        best_op = None
        best_score = 0

        for op_id, info in COMMAND_MANIFEST.items():
            score = sum(1 for trigger in info["triggers"] if trigger in q)
            if score > best_score:
                best_score = score
                best_op = op_id

        if best_op is None or best_score == 0:
            # Default to web search
            best_op = "web.search"

        return CommandPlan(
            reasoning=[f"Matched '{best_op}' by keyword analysis"],
            commands=[MappedCommand(
                operation=best_op,
                description=COMMAND_MANIFEST[best_op]["description"],
                parameters={"query": query},
            )],
            original_query=query,
        )
