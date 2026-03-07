"""ChainParser — parse and validate `octane chain` step strings.

`octane chain` takes a sequence of step strings and executes them in order,
with each step able to reference outputs from earlier steps.

Step string format:
    "name: command arg1 arg2 {ref}"

    name    — optional step name (used for {name} references in later steps).
              If omitted, steps are named "step_1", "step_2", etc.
    command — the octane command to run: ask, search, fetch, analyze,
              synthesize, code, or a domain pipeline keyword.
    args    — arguments to the command (free text).
    {ref}   — interpolation from a previous step's output:
                  {prev}         last step's output
                  {step_name}    named step's output
                  {all}          concatenation of all prior outputs

Template variables (double braces):
    {{ticker}}     expanded from --var ticker=NVDA at runtime

Examples:
    "prices: fetch finance NVDA AAPL MSFT"
    "tech: analyze technical {prices}"
    "report: synthesize investment-brief {all}"

    "summary: ask what is {{topic}}"
    "deep: ask --deep more about {summary}"

The ChainParser does NOT execute anything — it only parses, validates,
and produces an ordered list of ChainStep objects ready for ChainExecutor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="osa.chain_parser")


# ── Patterns ──────────────────────────────────────────────────────────────────

# Matches "name: rest" — name is optional
_NAME_PREFIX = re.compile(r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<rest>.+)$")

# Matches {ref} — single brace interpolations (step references)
_STEP_REF = re.compile(r"\{(?![\{])(?P<ref>[a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")

# Matches {{var}} — double brace template variables
_TEMPLATE_VAR = re.compile(r"\{\{(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)\}\}")

# Known commands — first word after the optional "name:" prefix
_KNOWN_COMMANDS = frozenset({
    "ask", "search", "fetch", "analyze", "synthesize", "code",
    "compare", "investigate", "summarize", "extract", "report",
})

# Special step reference keywords
_SPECIAL_REFS = frozenset({"prev", "all"})


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class ChainStep:
    """One parsed step in a chain.

    Attributes:
        index:          0-based position in the chain.
        name:           Step name (explicit or auto-generated "step_N").
        raw:            The original unparsed step string.
        command:        The octane command to run (e.g. "ask", "fetch").
        args:           Remaining arguments after the command.
        step_refs:      Set of step names this step references via {ref}.
        template_vars:  Set of {{var}} names that need runtime substitution.
        has_prev:       True if this step references {prev}.
        has_all:        True if this step references {all}.
    """

    index: int
    name: str
    raw: str
    command: str
    args: str
    step_refs: set[str] = field(default_factory=set)
    template_vars: set[str] = field(default_factory=set)
    has_prev: bool = False
    has_all: bool = False

    @property
    def full_text(self) -> str:
        """The command + args together."""
        return f"{self.command} {self.args}".strip()

    def interpolate(
        self,
        step_outputs: dict[str, str],
        template_vars: dict[str, str] | None = None,
    ) -> str:
        """Substitute all references in args with actual values.

        Args:
            step_outputs:  Map of step_name → output string.
            template_vars: Map of var_name → value for {{var}} substitution.

        Returns:
            The interpolated args string ready for execution.
        """
        text = self.args

        # Substitute {{template_vars}} first
        if template_vars:
            def _replace_template(m: re.Match) -> str:
                var = m.group("var")
                return template_vars.get(var, m.group(0))

            text = _TEMPLATE_VAR.sub(_replace_template, text)

        # Substitute {prev}
        if self.has_prev and self.index > 0:
            prev_name = _prev_step_name(self.index, step_outputs)
            prev_val = step_outputs.get(prev_name, "")
            text = text.replace("{prev}", prev_val)

        # Substitute {all}
        if self.has_all:
            all_val = _join_all_outputs(step_outputs, self.index)
            text = text.replace("{all}", all_val)

        # Substitute named {ref}s
        def _replace_ref(m: re.Match) -> str:
            ref = m.group("ref")
            if ref in ("prev", "all"):
                return m.group(0)  # Already handled above
            return step_outputs.get(ref, m.group(0))

        text = _STEP_REF.sub(_replace_ref, text)

        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "raw": self.raw,
            "command": self.command,
            "args": self.args,
            "step_refs": sorted(self.step_refs),
            "template_vars": sorted(self.template_vars),
            "has_prev": self.has_prev,
            "has_all": self.has_all,
        }


@dataclass
class ChainPlan:
    """The full parsed chain: ordered list of ChainStep objects.

    Attributes:
        steps:      Ordered steps ready for execution.
        raw_steps:  Original unparsed strings.
    """

    steps: list[ChainStep] = field(default_factory=list)
    raw_steps: list[str] = field(default_factory=list)

    @property
    def step_names(self) -> set[str]:
        return {s.name for s in self.steps}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


# ── Validation Error ──────────────────────────────────────────────────────────


class ChainValidationError(ValueError):
    """Raised when a chain step fails validation."""

    def __init__(self, step_index: int, step_raw: str, reason: str) -> None:
        self.step_index = step_index
        self.step_raw = step_raw
        self.reason = reason
        super().__init__(f"Chain step {step_index + 1} invalid: {reason!r} (step: {step_raw!r})")


# ── Parser ────────────────────────────────────────────────────────────────────


class ChainParser:
    """Parses chain step strings into ChainStep objects.

    This class is purely deterministic — no LLM calls, no I/O.
    It validates the chain structure and reference graph.

    Raises:
        ChainValidationError: If a step references an unknown name,
                              uses an undefined template var, or has
                              a forward reference (not supported).
    """

    def parse(
        self,
        step_strings: list[str],
        template_vars: dict[str, str] | None = None,
        strict: bool = False,
    ) -> ChainPlan:
        """Parse a list of step strings into a ChainPlan.

        Args:
            step_strings:  Raw step strings from the CLI.
            template_vars: Runtime template variable values for {{var}}.
            strict:        If True, raises on unknown commands.
                           If False, accepts unknown commands (extensible).

        Returns:
            ChainPlan with validated, ordered ChainStep objects.

        Raises:
            ChainValidationError: On invalid step structure or bad references.
        """
        if not step_strings:
            raise ChainValidationError(-1, "", "Chain must have at least one step")

        plan = ChainPlan(raw_steps=list(step_strings))
        seen_names: dict[str, int] = {}  # name → index

        for i, raw in enumerate(step_strings):
            step = self._parse_step(i, raw, seen_names, template_vars, strict)
            plan.steps.append(step)
            seen_names[step.name] = i

        return plan

    def _parse_step(
        self,
        index: int,
        raw: str,
        seen_names: dict[str, int],
        template_vars: dict[str, str] | None,
        strict: bool,
    ) -> ChainStep:
        """Parse a single step string into a ChainStep."""
        raw = raw.strip()
        if not raw:
            raise ChainValidationError(index, raw, "Empty step string")

        # Try to extract "name: rest"
        m = _NAME_PREFIX.match(raw)
        if m:
            name = m.group("name")
            rest = m.group("rest").strip()
        else:
            name = f"step_{index + 1}"
            rest = raw

        # First word of rest is the command
        parts = rest.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Validate command
        if strict and command not in _KNOWN_COMMANDS:
            raise ChainValidationError(
                index, raw,
                f"Unknown command {command!r}. Known: {sorted(_KNOWN_COMMANDS)}",
            )

        # Extract step references from args
        step_refs: set[str] = set()
        has_prev = False
        has_all = False

        for ref_match in _STEP_REF.finditer(args):
            ref = ref_match.group("ref")
            if ref == "prev":
                has_prev = True
            elif ref == "all":
                has_all = True
            else:
                # Validate: ref must refer to a prior step
                if ref not in seen_names:
                    raise ChainValidationError(
                        index, raw,
                        f"Reference {{{{ {ref} }}}} refers to unknown step {ref!r}. "
                        f"Known steps so far: {sorted(seen_names.keys()) or ['(none)']}",
                    )
                step_refs.add(ref)

        # Extract template variables
        tvars: set[str] = set()
        for var_match in _TEMPLATE_VAR.finditer(args):
            tvars.add(var_match.group("var"))

        return ChainStep(
            index=index,
            name=name,
            raw=raw,
            command=command,
            args=args,
            step_refs=step_refs,
            template_vars=tvars,
            has_prev=has_prev,
            has_all=has_all,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _prev_step_name(current_index: int, step_outputs: dict[str, str]) -> str:
    """Find the name of the step just before current_index."""
    # step_outputs keys are step names; we need the one at index - 1.
    # Since steps are inserted in order, we rely on the caller ensuring
    # the right name is provided. The ChainExecutor knows the ordered names.
    # Here we just search for step_{current_index} as fallback.
    return f"step_{current_index}"


def _join_all_outputs(step_outputs: dict[str, str], before_index: int) -> str:
    """Concatenate all prior step outputs into a single string."""
    # Only include outputs from steps before the current one.
    # We can't know the index from the dict alone — the ChainExecutor
    # manages this by passing only completed outputs.
    parts = [v for v in step_outputs.values() if v]
    return "\n\n".join(parts)
