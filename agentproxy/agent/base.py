"""BaseAgent ABC — all domain agents inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentproxy.graph.state import AgentState


class BaseAgent(ABC):
    """Abstract base for every agent subgraph.

    Subclasses must set ``name`` and implement ``build_graph``.
    The orchestrator calls ``compile()`` once and uses the resulting
    compiled graph as a subgraph node.
    """

    name: str = "base"

    def __init__(self, llm: BaseChatModel, memory: Any = None) -> None:
        self.llm = llm
        self.memory = memory

    # ── Subclass contract ────────────────────────────────────────
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Return a StateGraph (not yet compiled) defining this agent's flow."""
        ...

    # ── Compile helper ───────────────────────────────────────────
    def compile(self, **kwargs: Any) -> CompiledStateGraph:
        """Compile the subgraph — kwargs forwarded to StateGraph.compile."""
        return self.build_graph().compile(**kwargs)

    async def __call__(self, state: AgentState, **kwargs: Any) -> dict[str, Any]:
        """Run the compiled graph and return the final state diff."""
        compiled = self.compile(**kwargs)
        return await compiled.ainvoke(state)
