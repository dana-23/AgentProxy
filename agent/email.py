"""EmailAgent(BaseAgent)"""

from __future__ import annotations

from typing import Any

from agent.base import BaseAgent
from graph.state import AgentState                                                
from langgraph.graph import StateGraph, START, END

class EmailAgent(BaseAgent):
    name: str = "email"

    def build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("test_llm", self.test_llm)

        graph.add_edge(START, "test_llm")
        graph.add_edge("test_llm", END)

        return graph
    
    def test_llm(self, state: AgentState) -> dict:
        response = self.llm.invoke(state["messages"])
        return {"result": {"summary": response.content}}