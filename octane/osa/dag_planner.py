"""DAG Planner — decomposes compound queries into multi-node execution plans.

For simple queries the Decomposer's single-node path is sufficient.
For compound queries (e.g. "fetch data then write code"), the DAGPlanner
emits a multi-node DAG so agents run in the right order with real data.

LLM output format (constrained, no JSON):
    Each line: <step>. <agent>/<template> | <instruction>
    Dependency line: depends_on: <step>, <step>

Example:
    1. web/web_finance | MSFT capex last 5 years
    2. web/web_finance | GOOGL capex last 5 years
    3. code/code_generation | Chart the capex data
    depends_on: 1, 2

Fallback: single-node DAG using the existing keyword classifier.
"""

from __future__ import annotations

import asyncio
import re

import structlog

from octane.models.dag import TaskDAG, TaskNode

logger = structlog.get_logger().bind(component="osa.dag_planner")

# Agents and templates the planner is allowed to emit
_VALID_AGENTS = {"web", "code", "memory", "sysstat"}
_TEMPLATE_TO_SUBAGENT: dict[str, tuple[str, str]] = {
    "web_finance":     ("web",    "finance"),
    "web_news":        ("web",    "news"),
    "web_search":      ("web",    "search"),
    "code_generation": ("code",   "full_pipeline"),
    "memory_recall":   ("memory", "read"),
    "sysstat_health":  ("sysstat","monitor"),
}

# Patterns that signal a compound query needing multi-step planning
_COMPOUND_SIGNALS = re.compile(
    r"\b(then|after|using|based on|with the data|chart|graph|plot|visuali[sz]e|"
    r"compar\w+|analy[sz]\w+|and then|fetch.*code|search.*write|research.*implement)\b",
    re.IGNORECASE,
)

_PLANNER_SYSTEM = """\
You are a task planner for an agentic AI system. Break the user's query into 1–5 ordered steps.

Available agents/templates:
  web/web_finance     — fetch stock price or financial data for a specific ticker
  web/web_news        — search recent news and headlines
  web/web_search      — general web research or information lookup
  code/code_generation — write and execute Python code
  memory/memory_recall — recall something from earlier in the session

Output rules — STRICT:
- One step per line: "<N>. <agent>/<template> | <instruction>"
- If any step depends on earlier steps, add ONE line at the end: "depends_on: <N>, <N>"
- Use depends_on when a step needs results from a previous step (e.g. code needs fetched data)
- No extra text, no explanation, no markdown, no blank lines between steps
- Maximum 5 steps
- For financial queries about multiple companies, use one web/web_finance step per company

Example for "chart MSFT and GOOGL capex":
1. web/web_finance | MSFT total capex last 5 years
2. web/web_finance | GOOGL total capex last 5 years
3. code/code_generation | Create a bar chart comparing MSFT and GOOGL capex
depends_on: 3 needs 1, 2
"""


class DAGPlanner:
    """Plans multi-step DAGs for compound queries."""

    def __init__(self, bodega=None) -> None:
        self._bodega = bodega

    def is_compound(self, query: str) -> bool:
        """Quick heuristic: does this query need more than one agent?"""
        return bool(_COMPOUND_SIGNALS.search(query))

    async def plan(self, query: str) -> TaskDAG | None:
        """Attempt to build a multi-node DAG.

        Resolution order:
          1. Domain pipeline match  — keyword-based, zero latency, no LLM
          2. LLM planning           — full multi-step plan via Bodega
          3. Returns None           — caller falls back to single-node Decomposer

        Returns None if planning fails or produces a trivial 1-node plan.
        """
        from octane.osa.domain_pipelines import match_domain, build_dag as build_domain_dag

        # Step 1: Try domain pipeline (no LLM round-trip)
        domain = match_domain(query)
        if domain:
            domain_dag = build_domain_dag(query, domain)
            if domain_dag and len(domain_dag.nodes) > 1:
                logger.info(
                    "dag_domain_match",
                    domain=domain,
                    nodes=len(domain_dag.nodes),
                    query=query[:80],
                )
                return domain_dag

        # Step 2: LLM planning (requires Bodega)
        if self._bodega is None:
            return None

        try:
            raw = await asyncio.wait_for(
                self._bodega.chat_simple(
                    prompt=f'Query: "{query}"',
                    system=_PLANNER_SYSTEM,
                    temperature=0.0,
                    max_tokens=400,
                ),
                timeout=8.0,  # fast fallback to keyword decomposition if LLM stalls
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("dag_planner_llm_failed", error=str(exc))
            return None

        # Strip <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        dag = self._parse_plan(cleaned, query)

        if dag is None or len(dag.nodes) <= 1:
            logger.debug("dag_planner_trivial", raw=cleaned[:120])
            return None

        logger.info(
            "dag_planned",
            nodes=len(dag.nodes),
            waves=len(dag.execution_order()),
            query=query[:80],
        )
        return dag

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_plan(self, text: str, original_query: str) -> TaskDAG | None:
        """Parse the constrained LLM output into a TaskDAG."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        step_pattern = re.compile(
            r"^(\d+)\.\s+(\w+)/(\w+)\s*\|\s*(.+)$", re.IGNORECASE
        )
        depends_pattern = re.compile(
            r"depends_on\s*:?\s*(.+)", re.IGNORECASE
        )

        # Parse step lines
        steps: list[dict] = []  # {num, agent, template, instruction}
        depends_raw: str = ""

        for line in lines:
            m = step_pattern.match(line)
            if m:
                num, agent, template, instruction = m.groups()
                steps.append({
                    "num": int(num),
                    "agent": agent.lower(),
                    "template": template.lower(),
                    "instruction": instruction.strip(),
                })
                continue
            dm = depends_pattern.match(line)
            if dm:
                depends_raw = dm.group(1)

        if not steps:
            return None

        # Build nodes — assign stable task_ids indexed by step number
        num_to_id: dict[int, str] = {}
        nodes: list[TaskNode] = []

        for step in steps:
            agent, template = self._resolve_agent_template(step["agent"], step["template"])
            if agent is None:
                logger.warning("dag_planner_unknown_template",
                               agent=step["agent"], template=step["template"])
                continue

            node = TaskNode(
                agent=agent,
                instruction=step["instruction"],
                metadata={
                    "template": template,
                    "sub_agent": _TEMPLATE_TO_SUBAGENT.get(template, (agent, ""))[1],
                    "source": "dag_planner",
                    "step": str(step["num"]),
                },
            )
            num_to_id[step["num"]] = node.task_id
            nodes.append(node)

        if not nodes:
            return None

        # Apply depends_on — parse "3 needs 1, 2" or "3, 4" style
        if depends_raw:
            self._apply_dependencies(nodes, num_to_id, depends_raw, steps)

        return TaskDAG(
            original_query=original_query,
            reasoning=f"DAGPlanner: {len(nodes)}-node plan",
            nodes=nodes,
        )

    def _resolve_agent_template(
        self, agent_hint: str, template_hint: str
    ) -> tuple[str | None, str | None]:
        """Map agent/template hints from LLM to canonical values."""
        # Try exact template match first
        full = f"{agent_hint}_{template_hint}".lower()
        if full in _TEMPLATE_TO_SUBAGENT:
            agent, _ = _TEMPLATE_TO_SUBAGENT[full]
            return agent, full

        # Try just template_hint as a full template name
        if template_hint in _TEMPLATE_TO_SUBAGENT:
            agent, _ = _TEMPLATE_TO_SUBAGENT[template_hint]
            return agent, template_hint

        # Fuzzy: agent hint alone
        if agent_hint in _VALID_AGENTS:
            # Default sub-template per agent
            defaults = {
                "web": "web_search",
                "code": "code_generation",
                "memory": "memory_recall",
                "sysstat": "sysstat_health",
            }
            t = defaults[agent_hint]
            return agent_hint, t

        return None, None

    def _apply_dependencies(
        self,
        nodes: list[TaskNode],
        num_to_id: dict[int, str],
        depends_raw: str,
        steps: list[dict],
    ) -> None:
        """Parse the depends_on line and set TaskNode.depends_on fields."""
        # Extract all (dependent_step, [prerequisite_steps]) pairs
        # Handles formats:
        #   "3 needs 1, 2"  → step 3 depends on 1 and 2
        #   "3, 4"           → last step depends on all earlier
        #   "3"              → step 3 depends on everything before it

        # Try to find "X needs Y, Z" or "X: Y, Z" patterns
        needs_pattern = re.compile(r"(\d+)\s+(?:needs|requires|after)\s+([\d,\s]+)")
        matches = needs_pattern.findall(depends_raw)

        if matches:
            for dep_num_str, prereqs_str in matches:
                dep_num = int(dep_num_str)
                prereqs = [int(x.strip()) for x in prereqs_str.split(",") if x.strip().isdigit()]
                dep_id = num_to_id.get(dep_num)
                if dep_id is None:
                    continue
                for node in nodes:
                    if node.task_id == dep_id:
                        node.depends_on = [
                            num_to_id[p] for p in prereqs if p in num_to_id
                        ]
        else:
            # Fallback: extract all numbers, treat the largest as dependent on the rest
            nums = sorted({int(x) for x in re.findall(r"\d+", depends_raw)})
            if len(nums) >= 2:
                dep_num = nums[-1]
                prereqs = nums[:-1]
                dep_id = num_to_id.get(dep_num)
                if dep_id:
                    for node in nodes:
                        if node.task_id == dep_id:
                            node.depends_on = [
                                num_to_id[p] for p in prereqs if p in num_to_id
                            ]
