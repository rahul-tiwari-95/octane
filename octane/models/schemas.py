"""Core schemas — AgentRequest, AgentResponse, and shared types."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Standard input for every agent.

    Every agent receives this. OSA constructs it from the user query
    and enriches it with context from P&L, Memory, and decomposition.
    """

    query: str = Field(description="The user query or sub-task instruction")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID that ties all events for a single user query",
    )
    session_id: str = Field(
        default="cli",
        description="Session identifier for multi-turn context",
    )
    source: str = Field(
        default="user",
        description="Who created this request (user, osa, agent name)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: user profile, memory results, prior sub-task outputs",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Routing hints, model preferences, retry counts",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResponse(BaseModel):
    """Standard output from every agent.

    Every agent returns this. OSA collects these and feeds them to the Evaluator.
    """

    agent: str = Field(description="Name of the agent that produced this response")
    success: bool = Field(default=True, description="Whether the agent succeeded")
    output: str = Field(default="", description="Primary text output")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data (finance results, code output, memory chunks, etc.)",
    )
    error: str = Field(default="", description="Error message if success=False")
    correlation_id: str = Field(default="", description="Echoed from request")
    duration_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="model_used, tokens_in, tokens_out, sub_agent_traces, etc.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
