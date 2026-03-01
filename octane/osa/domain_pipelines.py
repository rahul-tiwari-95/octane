"""Domain pipeline templates for direct DAG construction without LLM decomposition.

When a query clearly matches a known domain (investment analysis, deep research,
comparative analysis, content creation), the DAGPlanner uses a pre-defined
template to build the task DAG instead of making an LLM call.

Benefits:
- Zero latency for domain match (pure regex, no LLM round-trip)
- Deterministic node structure (easier to test and debug)
- LLM still used for fallback on ambiguous queries

Pipelines
---------
investment_analysis  — ticker + fundamentals + news (parallel)
deep_research        — 4-angle web search + memory recall (parallel)
comparative_analysis — two-item fetch + code comparison (with dependency)
content_creation     — research + synthesis + write (sequential)
"""

from __future__ import annotations

import re

import structlog

from octane.models.dag import TaskDAG, TaskNode

logger = structlog.get_logger().bind(component="osa.domain_pipelines")


# ── Keyword matchers ──────────────────────────────────────────────────────────
# Scores: number of keyword hits in the query determines the winning pipeline.
# A pipeline must score ≥ _MIN_SCORE to fire.

_MIN_SCORE = 1

_PIPELINE_PATTERNS: dict[str, re.Pattern] = {
    "investment_analysis": re.compile(
        r"\b(invest|portfolio|stock|shares?|ticker|earnings|revenue|valuation|"
        r"P/E|dividend|capex|bull|bear|market.cap|financials?|annual.report|"
        r"growth|price.target|analyst)\b",
        re.IGNORECASE,
    ),
    "comparative_analysis": re.compile(
        r"\b(compare|comparison|versus|vs\.?|difference|better|best.between|"
        r"contrast|side.by.side|which.is.better|head.to.head)\b",
        re.IGNORECASE,
    ),
    "content_creation": re.compile(
        r"\b(write|draft|create|generate|compose|produce|blog.post|article|"
        r"essay|report|summary|write.up|write.a|document)\b",
        re.IGNORECASE,
    ),
    "deep_research": re.compile(
        r"\b(research|deep.dive|comprehensive|in.depth|full.analysis|survey|"
        r"investigate|explore|overview|breakdown|thorough)\b",
        re.IGNORECASE,
    ),
}

# Priority order: if two pipelines tie, earlier one in this list wins.
_PIPELINE_PRIORITY = [
    "comparative_analysis",  # most specific — two-item structure
    "investment_analysis",   # domain-specific (finance)
    "content_creation",      # write-focused
    "deep_research",         # broad research catch-all
]

# ── Variable extractors ───────────────────────────────────────────────────────

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_COMPANY_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "doordash": "DASH", "shopify": "SHOP", "salesforce": "CRM",
    "uber": "UBER", "lyft": "LYFT", "airbnb": "ABNB",
}

# Pattern 1: "compare/versus <A> and/vs/versus <B>"
_COMPARE_RE = re.compile(
    r"(?:compare|comparison between|versus|vs\.?)\s+"
    r"([A-Za-z][A-Za-z0-9]{0,19}(?:\s+[A-Za-z0-9]+){0,3}?)\s+"
    r"(?:and|vs\.?|versus)\s+"
    r"([A-Za-z][A-Za-z0-9]{0,19}(?:\s+[A-Za-z0-9]+){0,3})",
    re.IGNORECASE,
)
# Pattern 2: "<A> vs/versus <B>"
_COMPARE_RE2 = re.compile(
    r"([A-Za-z][A-Za-z0-9]{0,19}(?:\s+[A-Za-z0-9]+){0,2}?)\s+(?:vs\.?|versus)\s+([A-Za-z][A-Za-z0-9]{0,19}(?:\s+[A-Za-z0-9]+){0,2})",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset(
    "a an the and or but of in on at to for with by from is are was were be been "
    "being have has had do does did will would could should may might must can "
    "this that these those it its i we you he she they them their there here "
    "how why what when where which who whom whose what's how's why's".split()
)


def match_domain(query: str) -> str | None:
    """Return the best-matching domain pipeline name, or None if no match.

    Scoring: count keyword hits per pipeline; the pipeline with the highest
    score (≥ _MIN_SCORE) wins.  Ties are broken by _PIPELINE_PRIORITY order.
    """
    scores: dict[str, int] = {}
    for name, pat in _PIPELINE_PATTERNS.items():
        hits = pat.findall(query)
        if hits:
            scores[name] = len(hits)

    if not scores:
        return None

    max_score = max(scores.values())
    if max_score < _MIN_SCORE:
        return None

    # Among all pipelines at max_score, pick by priority
    for name in _PIPELINE_PRIORITY:
        if scores.get(name, 0) == max_score:
            return name

    return max(scores, key=lambda k: scores[k])


def build_dag(query: str, pipeline_name: str) -> TaskDAG | None:
    """Build a TaskDAG from a domain pipeline template.

    Returns None if variable extraction fails (caller falls back to LLM plan).
    """
    builders = {
        "investment_analysis": _build_investment_dag,
        "comparative_analysis": _build_comparative_dag,
        "content_creation": _build_content_creation_dag,
        "deep_research": _build_deep_research_dag,
    }
    builder = builders.get(pipeline_name)
    if builder is None:
        return None
    try:
        return builder(query)
    except Exception as exc:
        logger.warning("domain_pipeline_build_failed", pipeline=pipeline_name, error=str(exc))
        return None


# ── Pipeline builders ─────────────────────────────────────────────────────────


def _extract_ticker(query: str) -> str | None:
    """Best-effort ticker extraction: regex → company name map → None."""
    # Ignore common false positives
    false_pos = frozenset(["AI", "ML", "US", "UK", "GDP", "CEO", "CFO", "IPO", "ETF"])
    m = _TICKER_RE.findall(query)
    for t in m:
        if t not in false_pos and len(t) >= 2:
            return t
    # Company name lookup
    q_lower = query.lower()
    for name, ticker in _COMPANY_MAP.items():
        if name in q_lower:
            return ticker
    return None


def _extract_topic(query: str, remove_patterns: list[re.Pattern] | None = None) -> str:
    """Strip trigger keywords from the query to extract the core topic."""
    topic = query.strip()
    if remove_patterns:
        for pat in remove_patterns:
            topic = pat.sub("", topic).strip()
    # Remove leading articles / conjunctions
    topic = re.sub(r"^(a |an |the |write a |write an |draft a |create a |generate a )", "", topic, flags=re.IGNORECASE).strip()
    return topic or query


def _extract_compare_items(query: str) -> tuple[str, str] | None:
    """Extract two comparable items from the query.

    Handles: "compare X vs Y", "X versus Y", "X vs Y".
    Returns (item_a, item_b) or None if extraction fails.
    """
    for pat in (_COMPARE_RE, _COMPARE_RE2):
        m = pat.search(query)
        if m:
            a = m.group(1).strip().rstrip(" ,")
            b = m.group(2).strip().rstrip(" ,")
            if a and b and a.lower() != b.lower():
                return a, b
    return None


def _build_investment_dag(query: str) -> TaskDAG | None:
    """Two parallel nodes: finance data + recent news."""
    ticker = _extract_ticker(query)
    topic = ticker or _extract_topic(query)

    node_finance = TaskNode(
        agent="web",
        instruction=f"{topic} current price, market cap, PE ratio, and recent earnings",
        metadata={
            "template": "web_finance",
            "sub_agent": "finance",
            "source": "domain_pipeline",
            "pipeline": "investment_analysis",
        },
    )
    node_news = TaskNode(
        agent="web",
        instruction=f"{topic} latest analyst ratings, news, and market sentiment",
        metadata={
            "template": "web_news",
            "sub_agent": "news",
            "source": "domain_pipeline",
            "pipeline": "investment_analysis",
        },
    )

    logger.info("domain_pipeline_investment", ticker=topic, nodes=2)
    return TaskDAG(
        original_query=query,
        reasoning=f"Domain pipeline: investment_analysis for '{topic}'",
        nodes=[node_finance, node_news],
    )


def _build_comparative_dag(query: str) -> TaskDAG | None:
    """Three nodes: fetch item_a, fetch item_b, then code comparison (depends on both)."""
    items = _extract_compare_items(query)
    if items is None:
        logger.debug("comparative_pipeline_no_items", query=query[:80])
        return None   # fall back to LLM

    item_a, item_b = items

    node_a = TaskNode(
        agent="web",
        instruction=f"{item_a} — key metrics, price, performance, and recent highlights",
        metadata={
            "template": "web_finance",
            "sub_agent": "finance",
            "source": "domain_pipeline",
            "pipeline": "comparative_analysis",
        },
    )
    node_b = TaskNode(
        agent="web",
        instruction=f"{item_b} — key metrics, price, performance, and recent highlights",
        metadata={
            "template": "web_finance",
            "sub_agent": "finance",
            "source": "domain_pipeline",
            "pipeline": "comparative_analysis",
        },
    )
    node_compare = TaskNode(
        agent="code",
        instruction=(
            f"Using the data provided above, create a structured side-by-side comparison "
            f"of {item_a} and {item_b}. Include a summary table with key metrics."
        ),
        metadata={
            "template": "code_generation",
            "sub_agent": "full_pipeline",
            "source": "domain_pipeline",
            "pipeline": "comparative_analysis",
        },
        depends_on=[node_a.task_id, node_b.task_id],
    )

    logger.info("domain_pipeline_comparative", item_a=item_a, item_b=item_b, nodes=3)
    return TaskDAG(
        original_query=query,
        reasoning=f"Domain pipeline: comparative_analysis ({item_a} vs {item_b})",
        nodes=[node_a, node_b, node_compare],
    )


def _build_content_creation_dag(query: str) -> TaskDAG | None:
    """Three nodes: research (parallel web + news), then write code to produce document."""
    _WRITE_TRIGGERS = re.compile(
        r"\b(write|draft|create|generate|compose|produce|blog post|article|essay|report|summary)\b",
        re.IGNORECASE,
    )
    topic = _extract_topic(query, remove_patterns=[_WRITE_TRIGGERS])

    node_search = TaskNode(
        agent="web",
        instruction=f"{topic} — comprehensive background, key facts, and recent developments",
        metadata={
            "template": "web_search",
            "sub_agent": "search",
            "source": "domain_pipeline",
            "pipeline": "content_creation",
        },
    )
    node_news = TaskNode(
        agent="web",
        instruction=f"{topic} — latest news, expert opinions, and emerging trends",
        metadata={
            "template": "web_news",
            "sub_agent": "news",
            "source": "domain_pipeline",
            "pipeline": "content_creation",
        },
    )
    node_write = TaskNode(
        agent="code",
        instruction=(
            f"Using the research provided above, write a well-structured, "
            f"comprehensive piece about: {topic}. "
            f"Include an introduction, key sections with headers, and a conclusion."
        ),
        metadata={
            "template": "code_generation",
            "sub_agent": "full_pipeline",
            "source": "domain_pipeline",
            "pipeline": "content_creation",
        },
        depends_on=[node_search.task_id, node_news.task_id],
    )

    logger.info("domain_pipeline_content", topic=topic[:60], nodes=3)
    return TaskDAG(
        original_query=query,
        reasoning=f"Domain pipeline: content_creation for '{topic}'",
        nodes=[node_search, node_news, node_write],
    )


def _build_deep_research_dag(query: str) -> TaskDAG | None:
    """Four parallel web-search angles + memory recall."""
    _RESEARCH_TRIGGERS = re.compile(
        r"\b(research|deep.dive|comprehensive|in.depth|full.analysis|survey|"
        r"investigate|explore|overview|breakdown|thorough)\b",
        re.IGNORECASE,
    )
    topic = _extract_topic(query, remove_patterns=[_RESEARCH_TRIGGERS])

    angles = [
        ("latest developments and news", "web_news", "news"),
        ("key facts, background, and history", "web_search", "search"),
        ("expert analysis and opinions", "web_search", "search"),
        ("market data and quantitative metrics", "web_finance", "finance"),
    ]

    nodes: list[TaskNode] = []
    for angle_text, template, sub_agent in angles:
        nodes.append(TaskNode(
            agent="web",
            instruction=f"{topic} — {angle_text}",
            metadata={
                "template": template,
                "sub_agent": sub_agent,
                "source": "domain_pipeline",
                "pipeline": "deep_research",
            },
        ))

    # Memory recall: check for prior research on this topic
    nodes.append(TaskNode(
        agent="memory",
        instruction=f"Recall any prior research or notes about: {topic}",
        metadata={
            "template": "memory_recall",
            "sub_agent": "read",
            "source": "domain_pipeline",
            "pipeline": "deep_research",
        },
    ))

    logger.info("domain_pipeline_deep_research", topic=topic[:60], nodes=len(nodes))
    return TaskDAG(
        original_query=query,
        reasoning=f"Domain pipeline: deep_research for '{topic}' ({len(nodes)} angles)",
        nodes=nodes,
    )
