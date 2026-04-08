from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import OrchestratorState, AgentState
from agent.router_agent import RouterAgent, KNOWN_AGENTS

from agent.email_agent import EmailAgent
from agent.finance_agent import FinanceAgent


def build_agent_registry(llm):
    return {
        "email": EmailAgent(llm=llm).compile(),
        "finance": FinanceAgent(llm=llm).compile()
    }

def make_agent_node(compiled_agent):
    async def node(state: OrchestratorState):
        result = await compiled_agent.ainvoke({"messages": state["messages"]})
        return {
            "agent_results": [result],
            "messages": [result["messages"][-1]],
        }
    return node

def dispatch(state: OrchestratorState):
    return [
        Send(task["agent"], {"messages": state["messages"], "task": task})
        for task in state["plan"]
    ]

def build_orchestrator(llm):
    router = RouterAgent(llm=llm).compile()
    registry = build_agent_registry(llm)

    async def route(state: OrchestratorState):
        result = await router.ainvoke({"messages": state["messages"]})
        return {"plan": result["tasks"]}

    graph = StateGraph(OrchestratorState)

    graph.add_node("route", route)
    for name, compiled_agent in registry.items():
        graph.add_node(name, make_agent_node(compiled_agent))

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", dispatch, list(registry.keys()))
    for name in registry:
        graph.add_edge(name, END)

    return graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    import asyncio
    from config.settings import get_llm
    from langchain_core.messages import HumanMessage

    async def main():
        llm = get_llm()
        graph = build_orchestrator(llm)
        print(graph.get_graph().draw_ascii())
        print("\n" + "=" * 60 + "\n")

        response = await graph.ainvoke(
            {"messages": [HumanMessage(
                content="What did I buy today?"
            )]},
            config={"configurable": {"thread_id": "test"}},
        )
        last_msg = response["messages"][-1]
        content = last_msg.content
        if isinstance(content, list):
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
        else:
            text = content
        print(f"\nAgent: {text}")

    asyncio.run(main())