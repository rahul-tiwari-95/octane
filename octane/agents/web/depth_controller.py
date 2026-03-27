"""DepthController — convergence-based stopping for iterative deepening.

Tracks n-gram novelty across search rounds.  When new content repeats
what prior rounds already covered, deepening is stopped early to save
latency and LLM tokens.

Usage in WebAgent's deepening loop:

    controller = DepthController()
    controller.ingest(round_1_texts)           # baseline
    if controller.should_continue():
        # ... run follow-up round ...
        controller.ingest(round_2_texts)
        if controller.should_continue():
            # ... run another round ...

The novelty score is the fraction of new n-grams NOT seen in prior rounds.
When it drops below a threshold (default 0.15), deepening stops — the new
round isn't adding enough fresh information to justify the cost.
"""

from __future__ import annotations

import re
from collections import Counter

import structlog

logger = structlog.get_logger().bind(component="web.depth_controller")

# Defaults
_DEFAULT_NGRAM_SIZE = 3
_DEFAULT_NOVELTY_THRESHOLD = 0.15  # stop when < 15% of n-grams are new
_MAX_ROUNDS = 5


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class DepthController:
    """Track content novelty across deepening rounds.

    Args:
        ngram_size: Size of n-grams to track (default 3).
        novelty_threshold: Minimum novelty fraction to continue (default 0.15).
        max_rounds: Hard cap on total rounds regardless of novelty (default 5).
    """

    def __init__(
        self,
        ngram_size: int = _DEFAULT_NGRAM_SIZE,
        novelty_threshold: float = _DEFAULT_NOVELTY_THRESHOLD,
        max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        self._ngram_size = ngram_size
        self._novelty_threshold = novelty_threshold
        self._max_rounds = max_rounds
        self._seen: set[tuple[str, ...]] = set()
        self._round = 0
        self._novelty_history: list[float] = []

    def ingest(self, texts: list[str]) -> float:
        """Ingest a round of extracted texts and return the novelty score.

        Args:
            texts: List of extracted content strings from this round.

        Returns:
            Novelty score (0.0-1.0). 1.0 = all new; 0.0 = fully redundant.
        """
        self._round += 1
        all_new_ngrams: list[tuple[str, ...]] = []

        for text in texts:
            if not text:
                continue
            tokens = _tokenize(text)
            grams = _ngrams(tokens, self._ngram_size)
            all_new_ngrams.extend(grams)

        if not all_new_ngrams:
            self._novelty_history.append(0.0)
            return 0.0

        # Count how many n-grams are genuinely new
        new_count = sum(1 for g in all_new_ngrams if g not in self._seen)
        novelty = new_count / len(all_new_ngrams)

        # Add all n-grams to seen set
        self._seen.update(all_new_ngrams)

        self._novelty_history.append(novelty)
        logger.debug(
            "depth_controller_ingest",
            round=self._round,
            total_ngrams=len(all_new_ngrams),
            new_ngrams=new_count,
            novelty=round(novelty, 3),
            threshold=self._novelty_threshold,
        )
        return novelty

    def should_continue(self) -> bool:
        """Return True if another deepening round is worthwhile.

        Checks:
        1. Haven't exceeded max_rounds.
        2. Last round had enough novelty (above threshold).
        3. Novelty isn't trending sharply downward (2 consecutive drops).
        """
        if self._round >= self._max_rounds:
            logger.info("depth_controller_stop_max_rounds", round=self._round)
            return False

        if not self._novelty_history:
            return True  # no data yet — first round hasn't been ingested

        last_novelty = self._novelty_history[-1]
        if last_novelty < self._novelty_threshold:
            logger.info(
                "depth_controller_stop_low_novelty",
                round=self._round,
                novelty=round(last_novelty, 3),
                threshold=self._novelty_threshold,
            )
            return False

        # Check for rapidly declining novelty (two consecutive drops)
        if len(self._novelty_history) >= 3:
            n1, n2, n3 = self._novelty_history[-3:]
            if n1 > n2 > n3 and n3 < self._novelty_threshold * 1.5:
                logger.info(
                    "depth_controller_stop_declining",
                    round=self._round,
                    trend=[round(n, 3) for n in (n1, n2, n3)],
                )
                return False

        return True

    @property
    def round_count(self) -> int:
        """Number of rounds ingested so far."""
        return self._round

    @property
    def novelty_history(self) -> list[float]:
        """Per-round novelty scores."""
        return list(self._novelty_history)

    @property
    def total_unique_ngrams(self) -> int:
        """Total unique n-grams seen across all rounds."""
        return len(self._seen)

    def summary(self) -> dict:
        """Return a summary dict for logging/tracing."""
        return {
            "rounds": self._round,
            "total_unique_ngrams": len(self._seen),
            "novelty_history": [round(n, 3) for n in self._novelty_history],
            "converged": not self.should_continue() if self._novelty_history else False,
        }
