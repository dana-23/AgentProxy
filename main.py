from graph.orchestrator import build_orchestrator
from agent.finance_agent import FinanceAgent
from interfaces.discord_server import DiscordServer
from config.settings import get_settings, get_llm

if __name__ == "__main__":
    settings = get_settings()
    graph = FinanceAgent(get_llm()).compile()
    server = DiscordServer(graph, debug=settings.debug)
    server.run(settings.discord_bot_token)