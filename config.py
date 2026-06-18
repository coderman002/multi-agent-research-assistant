"""
config.py
---------
Centralized configuration loader for the Multi-Agent Research Assistant.
Reads from environment variables (populated via .env file).
Never hard-code credentials here — use .env instead.
"""

import os
from dotenv import load_dotenv

# Load .env file into environment
load_dotenv()


class Config:
    """
    Application configuration loaded from environment variables.
    All sensitive values (API keys, endpoints) must live in .env.
    """

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Set USE_AZURE=true in .env to switch to Azure OpenAI
    USE_AZURE: bool = os.getenv("USE_AZURE", "false").lower() == "true"

    # Standard OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Azure OpenAI (used only when USE_AZURE=true)
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # ── Tavily Search ──────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # ── Agent Behaviour ────────────────────────────────────────────────────────
    MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
    REPORT_LANGUAGE: str = os.getenv("REPORT_LANGUAGE", "English")

    @classmethod
    def validate(cls) -> None:
        """Raise a clear error if required keys are missing."""
        errors = []
        if cls.USE_AZURE:
            if not cls.AZURE_OPENAI_API_KEY:
                errors.append("AZURE_OPENAI_API_KEY")
            if not cls.AZURE_OPENAI_ENDPOINT:
                errors.append("AZURE_OPENAI_ENDPOINT")
        else:
            if not cls.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY")
        if not cls.TAVILY_API_KEY:
            errors.append("TAVILY_API_KEY")
        if errors:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(errors)}\n"
                "Copy .env.example to .env and fill in the values."
            )


config = Config()
