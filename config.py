#!/usr/bin/env python3
"""
Configuration management for the extraction system.

Handles API provider settings, model selection, and credentials.
"""
import json
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "api_provider": "openai",         # "anthropic", "openai", or "ollama"
    "model": "gpt-4o-mini",
    "api_key": "",                    # user provides, or use env var

    # Ollama settings (for local LLM)
    "ollama_host": "http://localhost:11434",
    "ollama_model": "",               # requires explicit selection
    "ollama_timeout": 300,            # 5 minutes for slow models

    "requests_per_minute": 60,        # rate limiting (1 request/second)
    "retry_attempts": 2,
    "theme_color": "#00ff00"          # UI theme color (hex)
}


def load_config() -> dict:
    """
    Load configuration from config.json.
    Creates file with defaults if it doesn't exist.
    """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with defaults to handle new config options
        merged = {**DEFAULT_CONFIG, **config}
        return merged
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save configuration to config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_api_key(config: dict = None) -> str | None:
    """
    Get API key from config or environment variable.

    Priority:
    1. config["api_key"] if non-empty
    2. ANTHROPIC_API_KEY or OPENAI_API_KEY env var (based on provider)
    """
    if config is None:
        config = load_config()

    # Check config first
    if config.get("api_key"):
        return config["api_key"]

    # Fall back to environment variable
    provider = config.get("api_provider", "anthropic")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "openai":
        return os.environ.get("OPENAI_API_KEY")

    return None


def validate_config(config: dict = None) -> tuple[bool, str]:
    """
    Validate configuration is complete and usable.
    Returns (is_valid, error_message).
    """
    if config is None:
        config = load_config()

    # Check provider
    provider = config.get("api_provider")
    if provider not in ("anthropic", "openai", "ollama"):
        return False, f"Invalid api_provider: {provider}. Must be 'anthropic', 'openai', or 'ollama'"

    # Provider-specific validation
    if provider == "ollama":
        # Check Ollama model is specified
        if not config.get("ollama_model"):
            return False, "No Ollama model selected. Please choose a model in Settings."

        # Check Ollama connection
        from llm_provider import check_ollama_connection
        host = config.get("ollama_host", "http://localhost:11434")
        if not check_ollama_connection(host):
            return False, f"Cannot connect to Ollama at {host}. Is Ollama running?"

    else:
        # Cloud providers need API key
        api_key = get_api_key(config)
        if not api_key:
            env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            return False, f"No API key configured. Set 'api_key' in config.json or {env_var} environment variable"

        # Check model
        if not config.get("model"):
            return False, "No model specified in config"

    return True, ""


if __name__ == "__main__":
    # Show current config when run directly
    config = load_config()
    print("Current configuration:")
    print(json.dumps(config, indent=2))

    is_valid, error = validate_config(config)
    if is_valid:
        print("\nConfiguration is valid.")
        # Mask API key for display
        key = get_api_key(config)
        if key:
            print(f"API key: {key[:8]}...{key[-4:]}")
    else:
        print(f"\nConfiguration error: {error}")
