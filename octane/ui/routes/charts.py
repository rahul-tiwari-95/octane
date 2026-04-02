"""octane/ui/routes/charts.py — Research analytics chart data API.

Endpoints for research activity, source distribution, and trust score
charts in Mission Control.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger("octane.ui.charts")

router = APIRouter(tags=["charts"])


@router.get("/charts/source-distribution")
async def source_distribution():
    """Source type breakdown of all extractions (arXiv, YouTube, web, PDF)."""
    try:
        from octane.tools.pg_client import PgClient

        pg = PgClient()
        rows = await pg.fetch(
            "SELECT source_type, COUNT(*) as cnt FROM extracted_documents GROUP BY source_type"
        )
        sources = [{"source": r["source_type"], "count": r["cnt"]} for r in rows]
        total = sum(s["count"] for s in sources)
        for s in sources:
            s["percentage"] = round(s["count"] / total * 100, 1) if total else 0

        return {"sources": sources, "total": total}
    except Exception as exc:
        logger.warning("source_distribution failed: %s", exc)
        return {"sources": [], "total": 0}


@router.get("/charts/research-activity")
async def research_activity():
    """Extractions per day for the last 30 days — for histogram."""
    try:
        from octane.tools.pg_client import PgClient

        pg = PgClient()
        rows = await pg.fetch(
            """
            SELECT DATE(created_at) as day, COUNT(*) as cnt
            FROM extracted_documents
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY day
            """
        )
        activity = [
            {"date": r["day"].isoformat(), "count": r["cnt"]}
            for r in rows
        ]
        return {"activity": activity, "days": len(activity)}
    except Exception as exc:
        logger.warning("research_activity failed: %s", exc)
        return {"activity": [], "days": 0}


@router.get("/charts/trust-scores")
async def trust_scores():
    """Distribution of reliability/trust scores across stored content."""
    try:
        from octane.tools.pg_client import PgClient

        pg = PgClient()
        rows = await pg.fetch(
            """
            SELECT
              CASE
                WHEN reliability_score >= 0.8 THEN 'high'
                WHEN reliability_score >= 0.5 THEN 'medium'
                WHEN reliability_score >= 0.2 THEN 'low'
                ELSE 'unknown'
              END as tier,
              COUNT(*) as cnt
            FROM extracted_documents
            WHERE reliability_score IS NOT NULL
            GROUP BY tier
            """
        )
        tiers = {r["tier"]: r["cnt"] for r in rows}
        return {
            "distribution": tiers,
            "total": sum(tiers.values()),
        }
    except Exception as exc:
        logger.warning("trust_scores failed: %s", exc)
        return {"distribution": {}, "total": 0}


@router.get("/charts/extraction-stats")
async def extraction_stats():
    """Aggregate extraction stats for the research overview."""
    try:
        from octane.tools.pg_client import PgClient

        pg = PgClient()
        row = await pg.fetchrow(
            """
            SELECT
              COUNT(*) as total,
              COUNT(DISTINCT source_type) as source_types,
              SUM(word_count) as total_words,
              AVG(word_count) as avg_words,
              MAX(created_at) as latest
            FROM extracted_documents
            """
        )
        return {
            "total_extractions": row["total"] if row else 0,
            "source_types": row["source_types"] if row else 0,
            "total_words": row["total_words"] if row else 0,
            "avg_words": round(row["avg_words"], 0) if row and row["avg_words"] else 0,
            "latest": row["latest"].isoformat() if row and row["latest"] else None,
        }
    except Exception as exc:
        logger.warning("extraction_stats failed: %s", exc)
        return {
            "total_extractions": 0,
            "source_types": 0,
            "total_words": 0,
            "avg_words": 0,
            "latest": None,
        }
