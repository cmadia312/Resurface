"""
Unified LLM provider abstraction for Resurface.

Supports: OpenAI, Anthropic, Ollama (local)

This module provides a single call_llm() function that routes to the
appropriate provider based on config. For Ollama, it uses structured
output via the format parameter with Pydantic schemas.
"""
import time
from typing import Any
import httpx
from pydantic import BaseModel

from config import get_api_key


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 300.0  # 5 minutes for slow local models


# =============================================================================
# EXCEPTIONS
# =============================================================================

class OllamaError(Exception):
    """Raised when Ollama API returns an error."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to Ollama."""
    pass


# =============================================================================
# OLLAMA HELPER FUNCTIONS
# =============================================================================

def check_ollama_connection(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """
    Check if Ollama is running and accessible.

    Args:
        host: Ollama API host URL

    Returns:
        True if Ollama is running and responding
    """
    try:
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    except Exception:
        return False


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    """
    Get list of available Ollama model names.

    Args:
        host: Ollama API host URL

    Returns:
        List of model names (e.g., ["llama3.2:3b", "gemma3:4b"])

    Raises:
        OllamaConnectionError: If unable to connect to Ollama
    """
    try:
        response = httpx.get(f"{host}/api/tags", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        return [m["name"] for m in models]
    except httpx.ConnectError:
        raise OllamaConnectionError(f"Cannot connect to Ollama at {host}. Is Ollama running?")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama API error: {e.response.status_code}")
    except Exception as e:
        raise OllamaError(f"Error listing models: {e}")


def make_array_schema(item_schema: type[BaseModel]) -> dict:
    """
    Create a JSON schema for an array of items.

    Ollama's format parameter requires the full schema.
    For arrays, we need: {"type": "array", "items": {...item schema...}}

    Args:
        item_schema: Pydantic model class for array items

    Returns:
        JSON schema dict for array of that type
    """
    return {
        "type": "array",
        "items": item_schema.model_json_schema()
    }


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================

def _call_openai(
    messages: list[dict],
    config: dict,
    max_tokens: int = 2048,
    system_prompt: str | None = None
) -> str:
    """
    Call OpenAI API.

    Args:
        messages: List of message dicts with "role" and "content"
        config: Configuration dict with api_key, model
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt (added as first message)

    Returns:
        Response text from the model
    """
    import openai

    api_key = get_api_key(config)
    client = openai.OpenAI(api_key=api_key)

    # Build messages with optional system prompt
    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = client.chat.completions.create(
        model=config.get('model', 'gpt-4o-mini'),
        max_tokens=max_tokens,
        messages=openai_messages
    )

    return response.choices[0].message.content


def _call_anthropic(
    messages: list[dict],
    config: dict,
    max_tokens: int = 2048
) -> str:
    """
    Call Anthropic API.

    Args:
        messages: List of message dicts with "role" and "content"
        config: Configuration dict with api_key, model
        max_tokens: Maximum tokens in response

    Returns:
        Response text from the model
    """
    import anthropic

    api_key = get_api_key(config)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=config.get('model', 'claude-sonnet-4-20250514'),
        max_tokens=max_tokens,
        messages=messages
    )

    return response.content[0].text


def _call_ollama(
    messages: list[dict],
    config: dict,
    schema: type[BaseModel] | dict | None = None,
    max_tokens: int = 2048
) -> str:
    """
    Call Ollama API with optional structured output.

    Args:
        messages: List of message dicts with "role" and "content"
        config: Configuration dict with ollama_host, ollama_model, ollama_timeout
        schema: Optional Pydantic model or dict schema for structured output
        max_tokens: Maximum tokens in response (used in num_predict option)

    Returns:
        Response text from the model

    Raises:
        OllamaConnectionError: If unable to connect to Ollama
        OllamaError: If API returns an error
    """
    host = config.get("ollama_host", DEFAULT_OLLAMA_HOST)
    model = config.get("ollama_model", "")
    timeout = config.get("ollama_timeout", DEFAULT_OLLAMA_TIMEOUT)

    if not model:
        raise OllamaError("No Ollama model specified. Please select a model in Settings.")

    # Build request payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,  # Deterministic output for consistency
            "num_predict": max_tokens
        }
    }

    # Add format parameter if schema provided (for structured output)
    if schema is not None:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Convert Pydantic model to JSON schema
            payload["format"] = schema.model_json_schema()
        elif isinstance(schema, dict):
            # Use dict directly as JSON schema (e.g., array schema)
            payload["format"] = schema
        else:
            raise ValueError(f"Schema must be a Pydantic model or dict, got {type(schema)}")

    # Make request
    try:
        response = httpx.post(
            f"{host}/api/chat",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
    except httpx.ConnectError:
        raise OllamaConnectionError(f"Cannot connect to Ollama at {host}. Is Ollama running?")
    except httpx.TimeoutException:
        raise OllamaError(f"Request timed out after {timeout}s")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

    # Parse response
    try:
        data = response.json()
    except Exception as e:
        raise OllamaError(f"Failed to parse response JSON: {e}")

    # Extract message content
    message = data.get("message", {})
    content = message.get("content", "")

    if not content:
        raise OllamaError(f"Empty response from model. Response data: {data}")

    return content


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def call_llm(
    messages: list[dict],
    config: dict,
    schema: type[BaseModel] | dict | None = None,
    max_tokens: int = 2048,
    system_prompt: str | None = None
) -> str:
    """
    Unified LLM call that routes to the appropriate provider.

    This is the main entry point for all LLM calls in Resurface.

    Args:
        messages: List of message dicts with "role" and "content" keys
        config: Configuration dict containing provider settings
        schema: Optional Pydantic model or dict schema for structured output
                (only used by Ollama - cloud providers use string schemas in prompts)
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt (handling varies by provider)

    Returns:
        Response text from the model

    Raises:
        ValueError: If unknown provider specified
        OllamaConnectionError: If Ollama is not running
        OllamaError: If Ollama API returns an error
    """
    provider = config.get('api_provider', 'openai')

    if provider == 'openai':
        return _call_openai(messages, config, max_tokens, system_prompt)
    elif provider == 'anthropic':
        return _call_anthropic(messages, config, max_tokens)
    elif provider == 'ollama':
        return _call_ollama(messages, config, schema, max_tokens)
    else:
        raise ValueError(f"Unknown API provider: {provider}")


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing LLM Provider...")

    # Test Ollama connection
    print("\n1. Checking Ollama connection...")
    if check_ollama_connection():
        print("   Connected to Ollama!")

        print("\n2. Listing available models...")
        models = list_ollama_models()
        for model in models:
            print(f"   - {model}")

        if models:
            print(f"\n3. Testing chat with {models[0]}...")
            test_config = {
                "api_provider": "ollama",
                "ollama_model": models[0],
                "ollama_host": DEFAULT_OLLAMA_HOST
            }

            try:
                response = call_llm(
                    messages=[{"role": "user", "content": "Say 'hello' in one word."}],
                    config=test_config
                )
                print(f"   Response: {response}")
            except Exception as e:
                print(f"   Error: {e}")
    else:
        print("   Cannot connect to Ollama. Is it running?")

    print("\nTest complete!")
