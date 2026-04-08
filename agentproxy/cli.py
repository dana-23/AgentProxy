"""AgentProxy CLI — control your agent from the terminal."""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(
    name="agentproxy",
    help="AI-powered agent that automates daily tasks.",
    add_completion=False,
)


@app.command()
def chat(
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider (anthropic, google, openai, ollama)"),
    model: str = typer.Option(None, "--model", "-m", help="Model name to use"),
):
    """Start an interactive terminal chat session."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from agentproxy.config.logger import setup_logging
    from agentproxy.config.settings import get_llm
    from agentproxy.graph.orchestrator import build_orchestrator

    setup_logging()
    llm = get_llm(provider, model)
    graph = build_orchestrator(llm)
    config = {"configurable": {"thread_id": "cli-session"}}

    typer.echo("AgentProxy chat (type 'quit' to exit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        result = asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
        )

        last_msg = result["messages"][-1]
        content = last_msg.content
        if isinstance(content, list):
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
        else:
            text = content
        typer.echo(f"\nAgent: {text}\n")

    typer.echo("Goodbye!")


@app.command()
def discord(
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model name"),
    channel: str = typer.Option("agent-chat", "--channel", "-c", help="Discord channel to listen on"),
):
    """Start the Discord bot."""
    from agentproxy.config.settings import get_llm, get_settings
    from agentproxy.graph.orchestrator import build_orchestrator
    from agentproxy.interfaces.discord_server import DiscordServer

    settings = get_settings()
    llm = get_llm(provider, model)
    graph = build_orchestrator(llm)
    server = DiscordServer(graph, channel_name=channel, debug=settings.debug)
    server.run(settings.discord_bot_token)


@app.command()
def run(
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model name"),
    interface: str = typer.Option("chat", "--interface", "-i", help="Interface to start (chat, discord)"),
):
    """Start AgentProxy with the given interface (default: chat)."""
    ctx = typer.Context
    if interface == "discord":
        discord(provider=provider, model=model)
    else:
        chat(provider=provider, model=model)


@app.command()
def config():
    """Show current configuration."""
    from agentproxy.config.settings import get_settings

    settings = get_settings()
    typer.echo(f"App:      {settings.app_name}")
    typer.echo(f"Provider: {settings.default_provider}")
    typer.echo(f"Model:    {settings.default_model}")
    typer.echo(f"Debug:    {settings.debug}")


def main():
    app()


if __name__ == "__main__":
    main()
