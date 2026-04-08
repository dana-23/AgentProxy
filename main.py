from agentproxy.graph.orchestrator import build_orchestrator
from agentproxy.agent.finance_agent import FinanceAgent
from agentproxy.interfaces.discord_server import DiscordServer
from agentproxy.config.settings import get_settings, get_llm

if __name__ == "__main__":
    settings = get_settings()
    graph = FinanceAgent(get_llm()).compile()
    server = DiscordServer(graph, debug=settings.debug)
    server.run(settings.discord_bot_token)