"""Chat Engine — the brain behind octane chat.

Session 37: Crown Jewel Chat.

Handles all five intent paths:
    CONVERSATION → direct LLM with persona (sub-second, no agents)
    COMMAND      → CommandMapper → execute → synthesize
    RECALL       → memory search → synthesize
    ANALYSIS     → full OSA pipeline (existing)
    WEB          → full OSA pipeline (existing)

Key features:
    - Persona-aware system prompts (name, personality from prefs)
    - Reasoning stream: shows plan in readable bullet points before execution
    - Steering window: 10-second pause after reasoning for user course-correction
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator

import structlog

from octane.osa.intent_gate import Intent, IntentGate
from octane.osa.command_mapper import CommandMapper, CommandPlan, MappedCommand
from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="chat_engine")


import re as _re

_CONTROL_TOKEN_RE = _re.compile(r"<\|[^|]*\|>")


async def _strip_think_stream(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Filter out <think>…</think> blocks and model control tokens from a streaming async iterator.

    Stateful: tracks whether we're inside a thinking block so that chunks
    arriving between split <think> and </think> tags are suppressed.
    Also strips leaked control tokens like <|im_end|>, <|endoftext|>, etc.
    """
    inside_think = False
    async for chunk in stream:
        if inside_think:
            # Look for the closing tag in this chunk
            if "</think>" in chunk:
                # Take only the text after </think>
                _, _, after = chunk.partition("</think>")
                inside_think = False
                after = _CONTROL_TOKEN_RE.sub("", after)
                if after:
                    yield after
            # else: still inside thinking block, suppress entire chunk
            continue

        # Not inside a think block
        if "<think>" in chunk:
            before, _, rest = chunk.partition("<think>")
            before = _CONTROL_TOKEN_RE.sub("", before)
            if before:
                yield before
            # Check if </think> also appears in the same chunk
            if "</think>" in rest:
                _, _, after = rest.partition("</think>")
                after = _CONTROL_TOKEN_RE.sub("", after)
                if after:
                    yield after
            else:
                inside_think = True
        else:
            chunk = _CONTROL_TOKEN_RE.sub("", chunk)
            if chunk:
                yield chunk


def build_persona_prompt(
    assistant_name: str = "octane",
    personality: str = "helpful, direct, and knowledgeable",
    user_name: str | None = None,
) -> str:
    """Build a persona-aware system prompt for conversational mode."""
    name = assistant_name.capitalize()
    lines = [
        f"You are {name}, a personal AI assistant.",
        f"Your personality: {personality}.",
        "",
        "You have access to a powerful knowledge system called Octane that can:",
        "- Search the web, fetch stock prices, get news",
        "- Research topics in depth, compare items side-by-side",
        "- Remember and recall past conversations and stored knowledge",
        "- Manage a financial portfolio, extract articles, generate code",
        "",
        "But right now you're in conversational mode — just be yourself.",
        "Be natural, warm, and conversational. Keep responses concise unless",
        "the user wants to go deeper. Don't be robotic.",
        "",
        "If the user asks you to do something that requires tools or data,",
        "tell them naturally that you'll look into it (the system will handle routing).",
    ]
    if user_name:
        lines.append(f"\nThe user's name is {user_name}.")

    return "\n".join(lines)


def build_command_synthesis_prompt(
    assistant_name: str = "octane",
    personality: str = "helpful, direct, and knowledgeable",
) -> str:
    """System prompt for synthesizing command results conversationally."""
    name = assistant_name.capitalize()
    return (
        f"You are {name}, a personal AI assistant. "
        f"Your personality: {personality}.\n\n"
        "You just executed some operations on behalf of the user. "
        "Present the results naturally and conversationally — don't just dump data.\n"
        "Highlight what matters. If there are numbers, give context.\n"
        "Be concise but insightful. If something looks notable (big gain, unusual data), "
        "call it out."
    )


class ChatEngine:
    """Core chat engine with intent-based routing and persona support."""

    def __init__(
        self,
        bodega=None,
        orchestrator=None,
        persona: dict | None = None,
    ) -> None:
        self._bodega = bodega
        self._orchestrator = orchestrator
        self._gate = IntentGate(bodega=bodega)
        self._mapper = CommandMapper(bodega=bodega)
        self._persona = persona or {}

    @property
    def assistant_name(self) -> str:
        return self._persona.get("assistant_name", "octane")

    @property
    def personality(self) -> str:
        return self._persona.get("assistant_personality", "helpful, direct, and knowledgeable")

    def update_persona(self, persona: dict) -> None:
        self._persona = persona

    # ── Follow-up query rewriter ─────────────────────────────────────────────

    # Patterns that indicate the query references previous context.
    _FOLLOWUP_SIGNALS = re.compile(
        r"^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|try again|"
        r"yes .{1,30}|yeah .{1,30})\s*$|"
        r"\b(that|this|those|it|them|the same)\b",
        re.IGNORECASE,
    )

    async def _rewrite_followup(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Rewrite a follow-up query into a self-contained query using
        conversation history.  Returns the original query unchanged if
        rewriting is not needed or not possible.

        Example:
            History: "what's the latest on Iran Israel war?"
            Query:   "yes find latest real time info"
            Result:  "find the latest real time info about the Iran Israel war"
        """
        # Only rewrite if there's history to reference and query is short
        if not conversation_history or len(query.split()) > 15:
            return query

        # Only rewrite if the query has follow-up signals
        if not self._FOLLOWUP_SIGNALS.search(query):
            return query

        # No LLM available — can't rewrite
        if self._bodega is None:
            return query

        # Build context from recent turns
        recent = conversation_history[-4:]
        context_parts = []
        for entry in recent:
            role = "User" if entry["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {entry['content'][:200]}")
        context = "\n".join(context_parts)

        system = (
            "You are a query rewriter. Given a conversation and a follow-up message, "
            "rewrite the follow-up into a self-contained query that makes sense without "
            "the conversation context.\n\n"
            "Rules:\n"
            "- Keep the rewritten query concise and natural\n"
            "- Preserve the user's intent (search, research, find news, etc.)\n"
            "- Replace pronouns (it, that, this) with the actual subject from context\n"
            "- If the follow-up is already self-contained, return it unchanged\n"
            "- Output ONLY the rewritten query, nothing else"
        )
        prompt = (
            f"Conversation:\n{context}\n\n"
            f"Follow-up message: \"{query}\"\n\n"
            f"Rewritten query:"
        )

        try:
            rewritten = await asyncio.wait_for(
                self._bodega.chat_simple(
                    prompt=prompt,
                    system=system,
                    tier=ModelTier.REASON,
                    temperature=0.1,
                    max_tokens=100,
                ),
                timeout=5.0,
            )
            # Strip thinking tags
            if "</think>" in rewritten:
                _, _, rewritten = rewritten.partition("</think>")
            rewritten = rewritten.strip().strip('"').strip()

            if rewritten and len(rewritten) > 3:
                logger.info(
                    "query_rewritten",
                    original=query[:60],
                    rewritten=rewritten[:80],
                )
                return rewritten
        except Exception as exc:
            logger.debug("query_rewrite_failed", error=str(exc))

        return query

    # ── Main entry point ─────────────────────────────────────────────────────

    async def respond(
        self,
        query: str,
        session_id: str,
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Route and respond to a user message. Yields streaming chunks."""

        intent, reasoning = await self._gate.classify(query, conversation_history)

        logger.info(
            "intent_classified",
            intent=intent.value,
            reasoning=reasoning,
            query=query[:80],
        )

        # For action intents, rewrite follow-up queries so they are
        # self-contained (e.g. "yes do it" → "find latest news on Iran war").
        if intent in (Intent.COMMAND, Intent.WEB, Intent.ANALYSIS):
            query = await self._rewrite_followup(query, conversation_history)

        if intent == Intent.CONVERSATION:
            async for chunk in self._handle_conversation(query, conversation_history):
                yield chunk

        elif intent == Intent.COMMAND:
            async for chunk in self._handle_command(query, session_id, conversation_history):
                yield chunk

        elif intent == Intent.RECALL:
            async for chunk in self._handle_recall(query, session_id, conversation_history):
                yield chunk

        elif intent in (Intent.ANALYSIS, Intent.WEB):
            async for chunk in self._handle_osa(query, session_id, conversation_history, intent):
                yield chunk

    # ── CONVERSATION path ────────────────────────────────────────────────────
    # Direct LLM call with persona. No agents, no web search.

    async def _handle_conversation(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Fast conversational response — direct LLM, no agents."""
        system = build_persona_prompt(
            assistant_name=self.assistant_name,
            personality=self.personality,
        )

        # Build messages with conversation history
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        # Include last 8 turns for continuity
        for entry in conversation_history[-8:]:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": query})

        if self._bodega is None:
            yield "I'm having trouble connecting to my language model. Try again in a moment."
            return

        try:
            raw = self._bodega.chat_stream(
                prompt=query,
                system=system,
                tier=ModelTier.REASON,
                temperature=0.7,
                max_tokens=512,
            )
            async for chunk in _strip_think_stream(raw):
                yield chunk
        except Exception as exc:
            logger.warning("conversation_llm_failed", error=str(exc))
            yield "Hmm, I had trouble generating a response. Let me try again."

    # ── COMMAND path ─────────────────────────────────────────────────────────
    # NL → CommandMapper → reasoning stream → execution → synthesis

    async def _handle_command(
        self,
        query: str,
        session_id: str,
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Map NL to commands, show reasoning, execute, synthesize."""

        # Step 1: Map to commands
        plan = await self._mapper.map(query, conversation_history)

        # Step 2: Stream reasoning bullets
        for bullet in plan.reasoning:
            yield f"  • {bullet}\n"

        if plan.commands:
            cmd_summary = ", ".join(c.operation for c in plan.commands)
            yield f"\n  [running: {cmd_summary}]\n\n"

        # Step 3: Execute commands
        results = await self._execute_commands(plan, session_id, original_query=query)

        # Step 4: Synthesize results conversationally.
        # If ALL results are pre-synthesized (came from OSA evaluator),
        # skip the redundant second LLM synthesis — just stream them.
        if results:
            all_pre_synth = all(r.get("pre_synthesized") for r in results if r["success"])
            if all_pre_synth:
                for r in results:
                    if r["success"] and r.get("output"):
                        yield r["output"]
            else:
                async for chunk in self._synthesize_results(query, results, conversation_history):
                    yield chunk
        else:
            yield "I tried to run that but didn't get any results. Could you rephrase?"

    async def _execute_commands(
        self,
        plan: CommandPlan,
        session_id: str,
        original_query: str = "",
    ) -> list[dict]:
        """Execute mapped commands and return their results."""
        results = []

        for cmd in plan.commands:
            try:
                result = await self._execute_single(cmd, session_id, original_query=original_query)
                # OSA-mapped operations return evaluator-synthesized text;
                # mark them so _handle_command skips redundant synthesis.
                osa_ops = {"web.search", "web.news", "web.finance", "code.generate", "system.health"}
                results.append({
                    "operation": cmd.operation,
                    "description": cmd.description,
                    "success": True,
                    "output": result,
                    "pre_synthesized": cmd.operation in osa_ops,
                })
            except Exception as exc:
                logger.warning(
                    "command_execution_failed",
                    operation=cmd.operation,
                    error=str(exc),
                )
                results.append({
                    "operation": cmd.operation,
                    "description": cmd.description,
                    "success": False,
                    "error": str(exc),
                })

        return results

    async def _execute_single(self, cmd: MappedCommand, session_id: str, original_query: str = "") -> str:
        """Execute a single mapped command and return its output as text.

        For operations that map to existing OSA agent capabilities, we delegate
        to the Orchestrator. For simpler operations, we call the underlying
        functions directly.
        """
        op = cmd.operation
        params = cmd.parameters

        # Operations that go through OSA agents (web search, news, finance, code)
        osa_mapping = {
            "web.search":       ("web", "search"),
            "web.news":         ("web", "news"),
            "web.finance":      ("web", "finance"),
            "code.generate":    ("code", "full_pipeline"),
            "system.health":    ("sysstat", "monitor"),
        }

        if op in osa_mapping and self._orchestrator:
            agent_name, sub_agent = osa_mapping[op]
            # Prefer extracted params, but fall back to original user query
            # so the DAG task contains the user's actual intent.
            query = params.get("query", params.get("topic", ""))
            if not query or len(query.split()) <= 3:
                query = original_query or cmd.description

            from octane.models.dag import TaskDAG, TaskNode
            dag = TaskDAG(
                original_query=query,
                reasoning=f"Command mapper: {cmd.description}",
                nodes=[TaskNode(
                    agent=agent_name,
                    instruction=query,
                    metadata={"sub_agent": sub_agent, "template": f"{agent_name}_{sub_agent}"},
                )],
            )

            parts = []
            async for chunk in self._orchestrator.run_from_dag(
                dag, query, session_id=session_id,
            ):
                parts.append(chunk)
            return "".join(parts)

        # Research operations
        if op == "research.start":
            topic = params.get("topic", cmd.description)
            query = f"research {topic}"
            if self._orchestrator:
                from octane.models.dag import TaskDAG, TaskNode
                dag = TaskDAG(
                    original_query=query,
                    reasoning=f"Research: {topic}",
                    nodes=[TaskNode(
                        agent="web",
                        instruction=f"Deep research on: {topic}",
                        metadata={"sub_agent": "search", "template": "web_search", "deep": True},
                    )],
                )
                parts = []
                async for chunk in self._orchestrator.run_from_dag(dag, query, session_id=session_id):
                    parts.append(chunk)
                return "".join(parts)
            return f"Research on '{topic}' requires the search pipeline."

        if op == "research.compare":
            items = params.get("items", [])
            criteria = params.get("criteria", "")
            if isinstance(items, list) and len(items) >= 2:
                query = f"compare {' vs '.join(items)}"
                if criteria:
                    query += f" in terms of {criteria}"
            else:
                query = cmd.description
            if self._orchestrator:
                from octane.models.dag import TaskDAG, TaskNode
                dag = TaskDAG(
                    original_query=query,
                    reasoning=f"Comparison: {query}",
                    nodes=[TaskNode(
                        agent="web",
                        instruction=query,
                        metadata={"sub_agent": "search", "template": "web_search"},
                    )],
                )
                parts = []
                async for chunk in self._orchestrator.run_from_dag(dag, query, session_id=session_id):
                    parts.append(chunk)
                return "".join(parts)

        # Portfolio operations
        if op == "portfolio.show":
            try:
                from octane.cli.portfolio import _portfolio_show_core
                return await _portfolio_show_core(prices=params.get("prices", True))
            except ImportError:
                return "Portfolio module not available."
            except Exception as exc:
                return f"Could not fetch portfolio: {exc}"

        # Recall operations
        if op in ("recall.search", "recall.stats"):
            if self._orchestrator:
                memory_agent = self._orchestrator._get_memory_agent()
                if memory_agent and op == "recall.search":
                    result = await memory_agent.recall(session_id, params.get("query", ""))
                    return result or "No matching memories found."
            return "Memory search requires a connected session."

        # Stats/status operations
        if op == "system.stats":
            return "Use the full `octane stats` command for the stats dashboard."

        if op == "watch.status":
            return "Use the full `octane watch status` command for monitoring details."

        if op == "model.list":
            if self._bodega:
                try:
                    models = await self._bodega.loaded_models_summary()
                    if models:
                        lines = ["Loaded models:"]
                        for m in models:
                            lines.append(f"  • {m['id']} ({m['type']}, ctx: {m['context_length']})")
                        return "\n".join(lines)
                    return "No models currently loaded."
                except Exception:
                    return "Could not fetch model info."
            return "Bodega not connected."

        if op == "pref.show":
            return "I'll show your preferences — use `/pref` in chat or `octane pref show`."

        # Extract operations
        if op == "extract.url":
            url = params.get("url", "")
            return f"To extract content from a URL, use: octane extract url \"{url}\""

        if op == "extract.arxiv":
            q = params.get("query", "")
            return f"To search arXiv, use: octane extract search-arxiv \"{q}\""

        # Vault operations
        if op in ("vault.encrypt", "vault.decrypt"):
            return f"Vault operations require the CLI: octane vault {op.split('.')[1]} <file>"

        return f"Operation {op} acknowledged but not yet wired for chat execution."

    async def _synthesize_results(
        self,
        query: str,
        results: list[dict],
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Synthesize command results into a natural response."""
        if self._bodega is None:
            # No LLM — just dump results
            for r in results:
                if r["success"]:
                    yield r["output"]
                else:
                    yield f"⚠ {r['operation']} failed: {r.get('error', 'unknown')}"
            return

        system = build_command_synthesis_prompt(
            assistant_name=self.assistant_name,
            personality=self.personality,
        )

        result_text = "\n\n".join(
            f"[{r['operation']}] {'SUCCESS' if r['success'] else 'FAILED'}\n"
            + (r.get("output", "") if r["success"] else f"Error: {r.get('error', '')}")
            for r in results
        )

        prompt = (
            f'User asked: "{query}"\n\n'
            f"Results:\n{result_text}\n\n"
            f"Present these results naturally to the user."
        )

        try:
            raw = self._bodega.chat_stream(
                prompt=prompt,
                system=system,
                tier=ModelTier.REASON,
                temperature=0.4,
                max_tokens=1024,
            )
            async for chunk in _strip_think_stream(raw):
                yield chunk
        except Exception as exc:
            logger.warning("synthesis_failed", error=str(exc))
            for r in results:
                if r["success"]:
                    yield r["output"]

    # ── RECALL path ──────────────────────────────────────────────────────────

    async def _handle_recall(
        self,
        query: str,
        session_id: str,
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Search memory and synthesize results with persona."""
        yield f"  • Searching my memory...\n\n"

        prior_context = None
        if self._orchestrator:
            memory_agent = self._orchestrator._get_memory_agent()
            if memory_agent:
                prior_context = await memory_agent.recall(session_id, query)

        if not prior_context:
            yield (
                f"I don't have anything stored about that yet. "
                f"As we chat and you explore topics, I'll build up knowledge about your interests."
            )
            return

        # Synthesize recall results with persona
        if self._bodega:
            system = build_persona_prompt(
                assistant_name=self.assistant_name,
                personality=self.personality,
            )
            prompt = (
                f'User asked: "{query}"\n\n'
                f"Here's what I found in memory:\n{prior_context}\n\n"
                f"Share this with the user naturally, as if you're recalling a conversation."
            )
            try:
                raw = self._bodega.chat_stream(
                    prompt=prompt,
                    system=system,
                    tier=ModelTier.REASON,
                    temperature=0.4,
                    max_tokens=800,
                )
                async for chunk in _strip_think_stream(raw):
                    yield chunk
            except Exception:
                yield prior_context
        else:
            yield prior_context

    # ── OSA path (analysis + web) ────────────────────────────────────────────

    async def _handle_osa(
        self,
        query: str,
        session_id: str,
        conversation_history: list[dict[str, str]],
        intent: Intent,
    ) -> AsyncIterator[str]:
        """Delegate to the full OSA pipeline with visible DAG plan."""
        if self._orchestrator is None:
            yield "The search pipeline isn't available right now. Try again in a moment."
            return

        # Decompose first so we can show the plan
        try:
            dag = await self._orchestrator.decomposer.decompose(query)
        except Exception as exc:
            logger.warning("osa_decompose_failed", error=str(exc))
            yield "I had trouble planning that. Could you rephrase?"
            return

        # Show the plan as readable bullets
        if intent == Intent.ANALYSIS:
            yield "  • Deep analysis mode\n"
        else:
            yield "  • Looking that up\n"

        agents_used = list({n.agent for n in dag.nodes})
        if agents_used:
            yield f"  • Agents: {', '.join(agents_used)}\n"
        if len(dag.nodes) > 1:
            yield f"  • {len(dag.nodes)} tasks in pipeline\n"

        yield "\n"

        # Execute via run_from_dag (bypasses re-decomposition)
        async for chunk in self._orchestrator.run_from_dag(
            dag,
            query,
            session_id=session_id,
            conversation_history=conversation_history,
        ):
            yield chunk
