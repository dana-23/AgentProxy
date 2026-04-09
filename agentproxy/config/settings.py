from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # agentproxy/
BASE_DIR = PACKAGE_DIR.parent  # project root


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "AgentProxy"
    debug: bool = False

    # --- API Keys ---
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    discord_bot_token: str = ""

    # --- Default LLM ---
    default_provider: str = "google"
    default_model: str = "gemini2.5-flash"

    model_config = {"env_file": BASE_DIR / ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this instead of Settings() directly."""
    return Settings()


def get_llm(provider: str | None = None, model: str | None = None):
    """Return a LangChain ChatModel for the given provider and model.

    Falls back to settings defaults when args are omitted.

    Usage:
        llm = get_llm()
        llm = get_llm("google", "gemini-2.0-flash")
        llm = get_llm("anthropic", "claude-sonnet-4")
        llm = get_llm("openai", "gpt-4o")
        llm = get_llm("ollama", "llama3")
    """
    settings = get_settings()
    provider = provider or settings.default_provider
    model = model or settings.default_model

    match provider:
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                api_key=settings.anthropic_api_key,
            )
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.google_api_key,
            )
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=settings.openai_api_key,
            )
        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                base_url=settings.ollama_base_url,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    s = get_settings()
    print(f"App:      {s.app_name}")
    print(f"Provider: {s.default_provider}")
    print(f"Model:    {s.default_model}")
    print(f"Debug:    {s.debug}")
    print("\nSettings loaded successfully.")
