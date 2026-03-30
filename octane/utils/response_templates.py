"""Response template loader — user-customizable output formats.

Users can place ``.txt`` files in ``~/.octane/templates/`` to override the
default synthesis format for any Octane command:

    ~/.octane/templates/compare.txt     → octane compare
    ~/.octane/templates/investigate.txt → octane investigate
    ~/.octane/templates/news.txt        → octane news / web news synthesis
    ~/.octane/templates/ask.txt         → octane ask --recall
    ~/.octane/templates/search.txt      → web search synthesis
    ~/.octane/templates/chat.txt        → octane chat result synthesis
    ~/.octane/templates/evaluate.txt    → OSA evaluator synthesis

Each file contains plain-English instructions describing the desired response
format.  These are appended to the system prompt so the LLM follows the user's
preferred structure.

Example ``~/.octane/templates/compare.txt``::

    Respond in bullet points.  At the end, write a short poem about the comparison.

If no template file is found, the built-in default is used unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_TEMPLATES_DIR = Path.home() / ".octane" / "templates"

# Cache: template name -> (content | None).  None means "file checked, not found".
_cache: dict[str, Optional[str]] = {}


def load_template(name: str, *, quiet: bool = False) -> str | None:
    """Load a user response template by command name.

    Args:
        name:  Template key (e.g. ``"compare"``, ``"investigate"``).
               Maps to ``~/.octane/templates/{name}.txt``.
        quiet: If False (default), print a one-line notice to stderr when a
               custom template is loaded so the user knows.

    Returns:
        The template text, or *None* if the file doesn't exist.
    """
    if name in _cache:
        return _cache[name]

    path = _TEMPLATES_DIR / f"{name}.txt"
    if not path.is_file():
        _cache[name] = None
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        _cache[name] = None
        return None

    _cache[name] = text
    if not quiet:
        print(
            f"[octane] Using custom response template: {name}.txt",
            file=sys.stderr,
            flush=True,
        )
    return text


def apply_template(system_prompt: str, name: str, *, quiet: bool = False) -> str:
    """Append the user's response template to a system prompt if one exists.

    If ``~/.octane/templates/{name}.txt`` exists, its contents are appended to
    *system_prompt* under a clear heading.  Otherwise *system_prompt* is
    returned unmodified.
    """
    template = load_template(name, quiet=quiet)
    if template is None:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"=== User Response Format ===\n"
        f"The user has requested the following output style. "
        f"Follow these formatting instructions:\n{template}"
    )


def clear_cache() -> None:
    """Clear the in-process template cache (useful for tests)."""
    _cache.clear()
