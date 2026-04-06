""" A Router Agent - Allocate's work to subagents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List
import yaml

from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.base import BaseAgent
from config.settings import BASE_DIR
from graph.state import AgentState

prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts.yaml')

class AgentTask(BaseModel):
    agent: str
    subtask: str

class CallSubAgent(BaseModel):
    tasks: List[AgentTask]

class RouterAgent(BaseAgent):

    name: str = "router"

    def build_graph(self) -> StateGraph:
        structure_llm = self.llm.with_structured_output(CallSubAgent)

        with open(prompts_path, 'r') as file:
            data = yaml.safe_load(file)

        system = SystemMessage(
            content=data.get("agents", {}).get(f"{self.name}", {}).get('system')
        )

        def call_llm(state: AgentState) -> dict:
            messages = [system] + list(state["messages"])
            response = structure_llm.invoke(messages)
            return {"result": {"tasks": [{"agent": t.agent, "subtask": t.subtask} for t in response.tasks]}}   

        graph = StateGraph(AgentState)

        graph.add_node("call_llm", call_llm)

        graph.add_edge(START, "call_llm")
        graph.add_edge("call_llm", END)

        return graph
    

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from config.settings import get_llm

    llm = get_llm()
    supreme_agent = RouterAgent(llm=llm)
    result = asyncio.run(
        supreme_agent(state={"messages": [HumanMessage(
            content="Summarize my recent email and write a follow up email to john doe for his car's extended warrenty."
            )]}
        )
    )

    print(result["result"])