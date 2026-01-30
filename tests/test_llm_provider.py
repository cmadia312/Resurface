"""
Tests for llm_provider.py - LLM provider abstraction.
"""
import pytest
from pydantic import BaseModel

from llm_provider import (
    call_llm,
    check_ollama_connection,
    make_array_schema,
    OllamaError,
    OllamaConnectionError
)
from schemas import ProjectIdea


def test_make_array_schema():
    """Test creating array schema from Pydantic model."""
    schema = make_array_schema(ProjectIdea)
    
    assert schema["type"] == "array"
    assert "items" in schema
    assert isinstance(schema["items"], dict)


def test_call_llm_openai_mock(sample_config, monkeypatch):
    """Test calling OpenAI with mocked response."""
    def mock_openai_call(messages, config, max_tokens, system_prompt):
        return '{"result": "success"}'
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_openai", mock_openai_call)
    
    config = {**sample_config, "api_provider": "openai"}
    messages = [{"role": "user", "content": "Test"}]
    
    response = call_llm(messages, config)
    
    assert response == '{"result": "success"}'


def test_call_llm_anthropic_mock(sample_config, monkeypatch):
    """Test calling Anthropic with mocked response."""
    def mock_anthropic_call(messages, config, max_tokens):
        return '{"result": "success"}'
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_anthropic", mock_anthropic_call)
    
    config = {**sample_config, "api_provider": "anthropic"}
    messages = [{"role": "user", "content": "Test"}]
    
    response = call_llm(messages, config)
    
    assert response == '{"result": "success"}'


def test_call_llm_ollama_mock(sample_config, monkeypatch):
    """Test calling Ollama with mocked response."""
    def mock_ollama_call(messages, config, schema, max_tokens):
        return '{"result": "success"}'
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_ollama", mock_ollama_call)
    
    config = {
        **sample_config,
        "api_provider": "ollama",
        "ollama_model": "llama3.2:3b"
    }
    messages = [{"role": "user", "content": "Test"}]
    
    response = call_llm(messages, config)
    
    assert response == '{"result": "success"}'


def test_call_llm_invalid_provider(sample_config):
    """Test that invalid provider raises ValueError."""
    config = {**sample_config, "api_provider": "invalid_provider"}
    messages = [{"role": "user", "content": "Test"}]
    
    with pytest.raises(ValueError, match="Unknown API provider"):
        call_llm(messages, config)


def test_call_llm_with_schema(sample_config, monkeypatch):
    """Test that schema is passed to Ollama."""
    captured_schema = None
    
    def mock_ollama_call(messages, config, schema, max_tokens):
        nonlocal captured_schema
        captured_schema = schema
        return '{"result": "success"}'
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_ollama", mock_ollama_call)
    
    config = {
        **sample_config,
        "api_provider": "ollama",
        "ollama_model": "test-model"
    }
    messages = [{"role": "user", "content": "Test"}]
    
    call_llm(messages, config, schema=ProjectIdea)
    
    assert captured_schema is ProjectIdea


def test_call_llm_with_system_prompt_openai(sample_config, monkeypatch):
    """Test that system prompt is passed to OpenAI."""
    captured_system = None
    
    def mock_openai_call(messages, config, max_tokens, system_prompt):
        nonlocal captured_system
        captured_system = system_prompt
        return "response"
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_openai", mock_openai_call)
    
    config = {**sample_config, "api_provider": "openai"}
    messages = [{"role": "user", "content": "Test"}]
    
    call_llm(messages, config, system_prompt="You are a helpful assistant")
    
    assert captured_system == "You are a helpful assistant"


def test_check_ollama_connection_success(monkeypatch):
    """Test checking Ollama connection when available."""
    class MockResponse:
        status_code = 200
    
    def mock_get(url, timeout):
        return MockResponse()
    
    import httpx
    monkeypatch.setattr(httpx, "get", mock_get)
    
    assert check_ollama_connection() is True


def test_check_ollama_connection_failure(monkeypatch):
    """Test checking Ollama connection when unavailable."""
    import httpx
    
    def mock_get(url, timeout):
        raise httpx.ConnectError("Connection refused")
    
    monkeypatch.setattr(httpx, "get", mock_get)
    
    assert check_ollama_connection() is False


def test_check_ollama_connection_timeout(monkeypatch):
    """Test checking Ollama connection on timeout."""
    import httpx
    
    def mock_get(url, timeout):
        raise httpx.TimeoutException("Timed out")
    
    monkeypatch.setattr(httpx, "get", mock_get)
    
    assert check_ollama_connection() is False


def test_ollama_error_inheritance():
    """Test that OllamaConnectionError inherits from OllamaError."""
    assert issubclass(OllamaConnectionError, OllamaError)
    
    error = OllamaConnectionError("Test")
    assert isinstance(error, OllamaError)
    assert isinstance(error, Exception)


def test_call_llm_custom_max_tokens(sample_config, monkeypatch):
    """Test that custom max_tokens is passed through."""
    captured_tokens = None
    
    def mock_openai_call(messages, config, max_tokens, system_prompt):
        nonlocal captured_tokens
        captured_tokens = max_tokens
        return "response"
    
    import llm_provider
    monkeypatch.setattr(llm_provider, "_call_openai", mock_openai_call)
    
    config = {**sample_config, "api_provider": "openai"}
    messages = [{"role": "user", "content": "Test"}]
    
    call_llm(messages, config, max_tokens=4096)
    
    assert captured_tokens == 4096
