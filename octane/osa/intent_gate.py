"""Intent Gate — pre-OSA classifier for octane chat.

Session 37: Crown Jewel Chat.

The Intent Gate classifies every user message into one of five intents
BEFORE the full OSA pipeline runs, enabling fast paths for conversation,
recall, and command decomposition.

Intents:
    CONVERSATION — casual chat, greetings, opinions, banter.
                   Handled directly by LLM with persona prompt.
                   No agents, no web search. Sub-second.

    COMMAND      — user wants to do something Octane can do:
                   show portfolio, research a topic, compare models, etc.
                   Routed to CommandMapper → internal execution → synthesis.

    RECALL       — user references stored knowledge or personal context:
                   "what did I read about…", "my portfolio", "remember when…"
                   Queries Redis + Postgres memory tiers.

    ANALYSIS     — deep multi-step question requiring web + synthesis:
                   "compare NVDA vs AMD fundamentals", "investigate…"
                   Full OSA pipeline with DAG planning.

    WEB          — simple factual lookup needing the internet:
                   "what's AAPL trading at?", "latest news on…"
                   Full OSA pipeline (single-node).
"""

from __future__ import annotations

import asyncio
import enum
import re

import structlog

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="intent_gate")


class Intent(enum.Enum):
    CONVERSATION = "conversation"
    COMMAND = "command"
    RECALL = "recall"
    ANALYSIS = "analysis"
    WEB = "web"


# ── Fast exact-match bypass ──────────────────────────────────────────────────
# Pure greetings and acks — no LLM call needed.

_CONVERSATION_EXACT = frozenset({
    "hi", "hello", "hey", "hey!", "hi!", "hello!",
    "thanks", "thank you", "thx", "ty",
    "ok", "okay", "k", "kk", "got it", "sure", "cool", "great", "nice",
    "sounds good", "perfect", "awesome", "good morning", "good evening",
    "good afternoon", "morning", "evening", "afternoon",
    "bye", "goodbye", "cya", "ttyl",
    "yes", "no", "yep", "nope", "yup",
    "huh", "hmm", "haha", "lol", "lmao", "wow",
    "how are you", "whats up", "what's up", "sup",
    "good night", "gn", "gm",
    "yo", "bruh", "dude", "bro",
    "i see", "makes sense", "interesting",
    "no worries", "np", "nw", "all good",
    "what do you think", "tell me more",
})

# ── Conversational follow-up patterns ────────────────────────────────────────
# These are phrases that clearly reference the PREVIOUS assistant response.
# They should stay in conversation mode, never trigger web search.
_FOLLOW_UP_PATTERNS = re.compile(
    r"^("
    r"what do you mean(\s+(by|about|with)\s+that)?|"
    r"what does that mean|what did you mean|what do u mean|"
    r"what are you saying|what's that mean|what is that|"
    r"why\??$|why is that|why do you say that|why would you say that|"
    r"why not|why so|how come|how so|how is that|"
    r"can you explain(\s+that)?|explain that|elaborate|go on|continue|"
    r"really|seriously|for real|are you sure|you sure|"
    r"like what|such as|for example|example|"
    r"in what way|in what sense|meaning|"
    r"say more|say that again|come again|what was that|"
    r"and then|so what|what else|anything else|"
    r"you think so|do you think|is that right|is that so|"
    r"that's it|that's all|is that all|"
    r"but why|but how|but what|"
    r"wait what|hold on|what\??$"
    r")\?*!*$",
    re.IGNORECASE,
)

# ── Recall signals ───────────────────────────────────────────────────────────
_RECALL_PATTERNS = re.compile(
    r"\b("
    r"remember|recall|forgot|stored|saved|previously|"
    r"we discussed|told you|i mentioned|last time|"
    r"what did i|what have i|what did we|what have we|articles i|"
    r"my notes|my bookmarks|my history|"
    r"have i read|have i seen|i read about|we read about|i looked at|"
    r"earlier about|past session|you said|"
    r"what do you know about me"
    r")\b",
    re.IGNORECASE,
)

# ── Command signals ──────────────────────────────────────────────────────────
# These map natural language to Octane CLI capabilities.
_COMMAND_PATTERNS = re.compile(
    r"\b("
    r"show me|show my|check my|"
    r"run|execute|start|stop|scan|"
    r"import|export|extract|download|"
    r"portfolio|positions|holdings|"
    r"research|investigate|look into|look up|dig into|"
    r"compare|versus|vs\.?|head.to.head|"
    r"monitor|watch|track|"
    r"set my|change my|update my|"
    r"system status|health check|"
    r"search for|find me|find\b|get me|fetch|"
    r"latest news|news on|news about|news related|"
    r"latest update|update on|update about|updates on|"
    r"what.?s happening|whats happening|"
    r"stock.?market|stock.?price|"
    r"summarize|summary of|"
    r"encrypt|decrypt|vault"
    r")\b",
    re.IGNORECASE,
)

# ── Analysis signals (deep multi-step) ───────────────────────────────────────
_ANALYSIS_PATTERNS = re.compile(
    r"\b("
    r"analyze|analyse|deep dive|in.depth|thorough|comprehensive|"
    r"breakdown|investigate|evaluate|assess|"
    r"pros and cons|advantages|disadvantages|"
    r"compare .+ (?:with|to|vs|and|against) .+"
    r")\b",
    re.IGNORECASE,
)

# ── LLM classification prompt ────────────────────────────────────────────────
_GATE_SYSTEM = """\
You are an intent classifier for a personal AI assistant.
Given a user message, classify it as EXACTLY one of these intents:

conversation — casual chat, greeting, opinion, banter, personal question, or anything that doesn't need data/tools
command — user wants to perform an action: show portfolio, research something, extract content, compare items, check system status, import data
recall — user references something stored: past conversations, articles read, bookmarks, personal knowledge
analysis — deep multi-step research question needing web search + synthesis, comparisons requiring data
web — simple factual lookup needing internet: current prices, latest news, specific facts

Rules:
- If the user is just talking/chatting, ALWAYS choose conversation
- If the user wants to DO something (show, research, import, compare), choose command
- If the user references memory/past data, choose recall
- Only choose web for simple factual lookups
- Choose analysis only for complex multi-step research
- When in doubt between conversation and web, choose conversation
- Output ONLY the intent word, nothing else"""


class IntentGate:
    """Classifies user intent before the OSA pipeline."""

    def __init__(self, bodega=None) -> None:
        self._bodega = bodega

    async def classify(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[Intent, str]:
        """Classify a user query into an Intent.

        Returns (intent, reasoning).
        """
        q_lower = query.strip().lower().rstrip("!.,?")

        # 1. Exact match — instant, no LLM
        if q_lower in _CONVERSATION_EXACT:
            return Intent.CONVERSATION, "Greeting/ack — direct response"

        # 2. Conversational follow-up — references the previous response
        #    e.g. "what do you mean?", "why?", "can you explain?"
        if _FOLLOW_UP_PATTERNS.search(q_lower):
            return Intent.CONVERSATION, "Conversational follow-up"

        # 3. Very short messages (1-3 words, no action verbs) → conversation
        words = q_lower.split()
        if len(words) <= 3 and not _COMMAND_PATTERNS.search(query) and not _RECALL_PATTERNS.search(query):
            return Intent.CONVERSATION, "Short conversational message"

        # 4. Strong recall signals
        if _RECALL_PATTERNS.search(query):
            return Intent.RECALL, "Memory/recall reference detected"

        # 5. Analysis signals (check before command — "compare X vs Y" is analysis)
        if _ANALYSIS_PATTERNS.search(query):
            return Intent.ANALYSIS, "Deep analysis/comparison detected"

        # 6. Command signals
        if _COMMAND_PATTERNS.search(query):
            return Intent.COMMAND, "Action/command intent detected"

        # 7. LLM classification for ambiguous queries
        if self._bodega is not None:
            try:
                result = await asyncio.wait_for(
                    self._classify_with_llm(query, conversation_history),
                    timeout=5.0,
                )
                if result:
                    return result
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("intent_gate_llm_failed", error=str(exc))

        # 8. Fallback: if it looks like a question, check if it's a
        #    conversational follow-up first (requires history context).
        if any(query.strip().lower().startswith(w) for w in
               ("what", "who", "when", "where", "how", "why", "is ", "are ", "do ", "does ", "can ", "will ")):
            # If we're mid-conversation and this is a short question,
            # it's almost certainly a follow-up, not a web search.
            if conversation_history and len(words) <= 8:
                last = conversation_history[-1] if conversation_history else None
                if last and last.get("role") == "assistant":
                    return Intent.CONVERSATION, "Short follow-up question in conversation"
            return Intent.WEB, "Question detected — web lookup"

        # 9. Default: conversation (better to chat than to search)
        return Intent.CONVERSATION, "Default — treating as conversation"

    async def _classify_with_llm(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[Intent, str] | None:
        """Use LLM for ambiguous intent classification."""
        context = ""
        if conversation_history:
            recent = conversation_history[-4:]
            parts = []
            for entry in recent:
                role = "User" if entry["role"] == "user" else "Assistant"
                parts.append(f"{role}: {entry['content'][:150]}")
            context = f"\nRecent conversation:\n" + "\n".join(parts) + "\n\n"

        prompt = f'{context}User message: "{query}"'

        raw = await self._bodega.chat_simple(
            prompt=prompt,
            system=_GATE_SYSTEM,
            tier=ModelTier.FAST,
            temperature=0.0,
            max_tokens=10,
        )

        # Strip thinking tags if present
        if "</think>" in raw:
            _, _, raw = raw.partition("</think>")
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)        # Strip chat template tokens (e.g. <|im_end|>, <|eot_id|>, <|end|>)
        raw = re.sub(r"<\|[^|]*\|>", "", raw)
        intent_str = raw.strip().lower().strip('"\'.,;:\n ')

        intent_map = {
            "conversation": Intent.CONVERSATION,
            "command": Intent.COMMAND,
            "recall": Intent.RECALL,
            "analysis": Intent.ANALYSIS,
            "web": Intent.WEB,
        }

        intent = intent_map.get(intent_str)
        if intent is None:
            logger.warning("intent_gate_unknown", raw=raw[:50])
            return None

        return intent, f"LLM classified: {intent_str}"
