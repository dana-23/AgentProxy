from langgraph.checkpoint.memory import MemorySaver
from interfaces.discord_server import DiscordServer
from agent.email_agent import EmailAgent
from config.settings import get_settings, get_llm

if __name__ == "__main__":
    settings = get_settings()
    graph = EmailAgent(llm=get_llm()).compile(checkpointer=MemorySaver())
    server = DiscordServer(graph)
    server.run(settings.discord_bot_token)