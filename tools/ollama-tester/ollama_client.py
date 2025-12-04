"""
Ollama API client with structured output support.

Uses native Ollama API with `format` parameter for schema-constrained JSON output.
"""
import time
from typing import Any
import httpx
from pydantic import BaseModel


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 300.0  # 5 minutes for slow models


class OllamaError(Exception):
    """Raised when Ollama API returns an error."""
    pass


class OllamaConnectionError(Exception):
    """Raised when unable to connect to Ollama."""
    pass


def list_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[dict]:
    """
    Get available Ollama models.

    Returns:
        List of model info dicts with 'name', 'size', 'modified_at' keys.

    Raises:
        OllamaConnectionError: If unable to connect to Ollama.
    """
    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except httpx.ConnectError:
        raise OllamaConnectionError(f"Cannot connect to Ollama at {host}. Is Ollama running?")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama API error: {e.response.status_code}")
    except Exception as e:
        raise OllamaError(f"Unexpected error: {e}")


def get_model_names(host: str = DEFAULT_HOST) -> list[str]:
    """Get list of available model names."""
    models = list_models(host)
    return [m["name"] for m in models]


def check_connection(host: str = DEFAULT_HOST) -> bool:
    """Check if Ollama is running and accessible."""
    try:
        list_models(host, timeout=5.0)
        return True
    except (OllamaConnectionError, OllamaError):
        return False


def make_array_schema(item_schema: type[BaseModel]) -> dict:
    """
    Create a JSON schema for an array of items.

    Ollama's format parameter requires the full schema.
    For arrays, we need: {"type": "array", "items": {...item schema...}}
    """
    return {
        "type": "array",
        "items": item_schema.model_json_schema()
    }


def chat(
    model: str,
    messages: list[dict],
    schema: type[BaseModel] | dict | None = None,
    temperature: float = 0.0,
    host: str = DEFAULT_HOST,
    timeout: float = DEFAULT_TIMEOUT
) -> tuple[str, float]:
    """
    Send chat completion request to Ollama.

    Args:
        model: Model name (e.g., "llama3.2:3b")
        messages: List of message dicts with "role" and "content" keys
        schema: Optional Pydantic model or JSON schema dict for structured output
        temperature: Sampling temperature (0.0 for deterministic)
        host: Ollama API host
        timeout: Request timeout in seconds

    Returns:
        Tuple of (response_text, duration_seconds)

    Raises:
        OllamaConnectionError: If unable to connect
        OllamaError: If API returns an error
    """
    # Build request payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    # Add format parameter if schema provided
    if schema is not None:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Convert Pydantic model to JSON schema
            payload["format"] = schema.model_json_schema()
        elif isinstance(schema, dict):
            # Use dict directly as JSON schema
            payload["format"] = schema
        else:
            raise ValueError(f"Schema must be a Pydantic model or dict, got {type(schema)}")

    # Make request
    start_time = time.time()

    try:
        response = httpx.post(
            f"{host}/api/chat",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
    except httpx.ConnectError:
        raise OllamaConnectionError(f"Cannot connect to Ollama at {host}")
    except httpx.TimeoutException:
        raise OllamaError(f"Request timed out after {timeout}s")
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

    duration = time.time() - start_time

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

    return content, duration


def chat_with_system(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel] | dict | None = None,
    temperature: float = 0.0,
    host: str = DEFAULT_HOST,
    timeout: float = DEFAULT_TIMEOUT
) -> tuple[str, float]:
    """
    Convenience function for system + user prompt pattern.

    Args:
        model: Model name
        system_prompt: System message content
        user_prompt: User message content
        schema: Optional schema for structured output
        temperature: Sampling temperature
        host: Ollama API host
        timeout: Request timeout

    Returns:
        Tuple of (response_text, duration_seconds)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return chat(model, messages, schema, temperature, host, timeout)


if __name__ == "__main__":
    # Test the client
    print("Testing Ollama client...")

    if not check_connection():
        print("ERROR: Cannot connect to Ollama. Is it running?")
        exit(1)

    print("Connected to Ollama!")

    models = get_model_names()
    print(f"Available models: {models}")

    if models:
        test_model = models[0]
        print(f"\nTesting chat with {test_model}...")

        response, duration = chat(
            model=test_model,
            messages=[{"role": "user", "content": "Say 'hello' in JSON format: {\"greeting\": \"...\"}"}],
            temperature=0.0
        )

        print(f"Response: {response}")
        print(f"Duration: {duration:.2f}s")
