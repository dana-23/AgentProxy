"""AgentProxy CLI — control your agent from the terminal."""

from __future__ import annotations

import asyncio
import uuid

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

theme = Theme(
    {
        "user": "bold cyan",
        "agent": "bold green",
        "info": "dim",
        "warn": "bold yellow",
    }
)
console = Console(theme=theme)

app = typer.Typer(
    name="agentproxy",
    help="AI-powered agent that automates daily tasks.",
    add_completion=False,
)


def _extract_text(result: dict) -> str:
    """Pull the text content from the last message in a graph result."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return next((b["text"] for b in content if b.get("type") == "text"), "")
    return content


def _show_help() -> None:
    console.print(
        Panel(
            "/help    — show this message\n"
            "/clear   — clear the screen\n"
            "/config  — show current settings\n"
            "/new     — start a new session\n"
            "/graph   — print the orchestrator graph\n"
            "/quit    — exit",
            title="[bold]Commands[/bold]",
            border_style="dim",
        )
    )


@app.command()
def chat(
    provider: str = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (anthropic, google, openai, ollama)",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model name to use"),
):
    """Start an interactive terminal chat session."""
    from langchain_core.messages import HumanMessage

    from agentproxy.config.logger import setup_logging
    from agentproxy.config.settings import get_llm, get_settings
    from agentproxy.graph.orchestrator import build_orchestrator

    setup_logging()
    settings = get_settings()
    llm = get_llm(provider, model)
    graph = build_orchestrator(llm)

    provider_display = provider or settings.default_provider
    model_display = model or settings.default_model

    console.print(
        Panel(
            f"[bold]AgentProxy[/bold]\n"
            f"[info]provider:[/info] {provider_display}  "
            f"[info]model:[/info] {model_display}\n"
            f"[info]Type /help for commands[/info]",
            border_style="cyan",
        )
    )

    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = console.input("[user]> [/user]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue

        # --- slash commands ---
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                _show_help()
            elif cmd == "/clear":
                console.clear()
            elif cmd == "/config":
                console.print(
                    Panel(
                        f"provider : {provider_display}\n"
                        f"model    : {model_display}\n"
                        f"session  : {thread_id}\n"
                        f"debug    : {settings.debug}",
                        title="[bold]Config[/bold]",
                        border_style="dim",
                    )
                )
            elif cmd == "/new":
                thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": thread_id}}
                console.print("[info]Started new session.[/info]")
            elif cmd == "/graph":
                console.print(graph.get_graph().draw_ascii())
            else:
                console.print(f"[warn]Unknown command: {cmd}[/warn]")
            continue

        # --- agent invocation ---
        with console.status("[info]thinking…[/info]", spinner="dots"):
            result = asyncio.run(
                graph.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
            )

        text = _extract_text(result)
        console.print()
        console.print(Markdown(text))
        console.print()

    console.print("[info]Goodbye![/info]")


@app.command()
def discord(
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model name"),
    channel: str = typer.Option(
        "agent-chat", "--channel", "-c", help="Discord channel to listen on"
    ),
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
    interface: str = typer.Option(
        "chat", "--interface", "-i", help="Interface to start (chat, discord)"
    ),
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
