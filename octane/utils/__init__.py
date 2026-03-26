"""Structured logging configuration using structlog."""

import structlog
from octane.config import settings


def setup_logging(level_override: str | None = None) -> None:
    """Configure structlog for Octane.

    Uses console renderer for development, JSON for production.

    Args:
        level_override: If provided, override settings.log_level (e.g. "debug"
            when --verbose is passed on the CLI).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Map log level name to numeric value
    level_name = (level_override or settings.log_level).lower()
    level_map = {"debug": 10, "info": 20, "warning": 30, "warn": 30, "error": 40, "critical": 50}
    log_level = level_map.get(level_name, 30)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger, optionally bound to a component name."""
    log = structlog.get_logger()
    if name:
        log = log.bind(component=name)
    return log
