"""Shared state schemas for the orchestrator and agent subgraphs."""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AnyMessage  # needed for get_type_hints on Python <3.12
from langgraph.graph import MessagesState, add_messages


# ── Agent-level state ────────────────────────────────────────────────
class AgentState(MessagesState):
    """
    State passed into every agent subgraph.

    Extends MessagesState (which provides a `messages` list with
    add-semantics) and adds fields the orchestrator injects.
    """
    
    task: dict[str, Any]
    result: dict[str, Any]


# ── Task descriptor ──────────────────────────────────────────────────
@dataclass
class Task:
    """A single unit of work the planner wants an agent to handle."""

    agent: str
    goal: str
    context: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


# ── Orchestrator-level state ─────────────────────────────────────────
class OrchestratorState(MessagesState):
    """
    Top-level state for the orchestrator graph.

    `agent_results` uses `operator.add` as a reducer so parallel
    agent branches can each append their result and LangGraph merges them.
    """

    intent: str
    entities: list[dict[str, Any]]
    memory_context: list[str]
    plan: list[Task]
    agent_results: Annotated[list[dict[str, Any]], operator.add]
    done: bool
