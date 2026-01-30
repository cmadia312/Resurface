"""
Tests for extractor.py - Conversation insight extraction.
"""
import json

import pytest

from extractor import (
    extract_conversation,
    format_conversation,
    parse_json_response
)


def test_format_conversation(sample_parsed_conversation):
    """Test formatting a parsed conversation for LLM prompt."""
    result = format_conversation(sample_parsed_conversation)
    
    assert "Title: Building a Task Manager" in result
    assert "Date: 2023-12-22" in result
    assert "USER:" in result or "ASSISTANT:" in result
    assert "task manager app" in result


def test_format_conversation_includes_all_messages(sample_parsed_conversation):
    """Test that all messages are included in formatted output."""
    result = format_conversation(sample_parsed_conversation)
    
    for msg in sample_parsed_conversation["messages"]:
        # Content should appear in formatted output
        assert msg["content"] in result


def test_format_conversation_handles_untitled():
    """Test formatting conversation with missing title."""
    conv = {
        "title": None,
        "created": "2023-12-22T12:00:00Z",
        "messages": [
            {"role": "user", "content": "Test"}
        ]
    }
    
    result = format_conversation(conv)
    # When title is None, it prints "None" literally
    assert "Title:" in result
    assert "Date: 2023-12-22" in result


def test_parse_json_response_valid_json():
    """Test parsing valid JSON response."""
    response = '{"project_ideas": [], "problems": []}'
    result = parse_json_response(response)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "project_ideas" in result


def test_parse_json_response_with_markdown():
    """Test parsing JSON wrapped in markdown code block."""
    response = """Here's the extraction:
```json
{
  "project_ideas": [{"idea": "Test", "motivation": "Testing", "detail_level": "vague"}],
  "problems": []
}
```
"""
    result = parse_json_response(response)
    
    assert result is not None
    assert len(result["project_ideas"]) == 1


def test_parse_json_response_with_extra_text():
    """Test parsing JSON with extra text before/after."""
    response = """Let me analyze this conversation.

{
  "project_ideas": [],
  "problems": [],
  "emotional_signals": {"tone": "neutral", "notes": ""}
}

Hope this helps!"""
    
    result = parse_json_response(response)
    
    assert result is not None
    assert "emotional_signals" in result


def test_parse_json_response_invalid():
    """Test handling of invalid JSON."""
    response = "This is not JSON at all"
    result = parse_json_response(response)
    
    assert result is None


def test_parse_json_response_malformed_braces():
    """Test handling of malformed JSON."""
    response = "{project_ideas: []"  # Missing closing brace, no quotes
    result = parse_json_response(response)
    
    assert result is None


def test_extract_conversation_with_mock(sample_parsed_conversation, sample_config, mock_llm_response, monkeypatch):
    """Test extraction with mocked LLM call."""
    # Mock the call_llm function
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        return mock_llm_response
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    result = extract_conversation(sample_parsed_conversation, sample_config)
    
    assert "project_ideas" in result
    assert len(result["project_ideas"]) > 0
    assert result["project_ideas"][0]["idea"] == "AI-powered note organizer"


def test_extract_conversation_handles_empty_response(sample_parsed_conversation, sample_config, monkeypatch):
    """Test extraction handles empty/invalid LLM response."""
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        return '{"empty": true, "reason": "Nothing extractable"}'
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    result = extract_conversation(sample_parsed_conversation, sample_config)
    
    assert result.get("empty") is True
    assert "reason" in result


def test_extract_conversation_handles_api_error(sample_parsed_conversation, sample_config, monkeypatch):
    """Test extraction handles LLM API errors."""
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        raise Exception("API connection failed")
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    result = extract_conversation(sample_parsed_conversation, sample_config)
    
    assert result.get("error") is True
    assert result.get("error_type") == "api_error"
    assert "API connection failed" in result.get("error_message", "")


def test_extract_conversation_retries_on_parse_failure(sample_parsed_conversation, sample_config, monkeypatch):
    """Test extraction retries when JSON parsing fails."""
    call_count = 0
    
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First call returns invalid JSON
            return "This is not valid JSON"
        else:
            # Subsequent calls return valid JSON
            return '{"project_ideas": [], "problems": [], "workflows": [], "tools_explored": [], "underlying_questions": [], "emotional_signals": {"tone": "neutral", "notes": ""}}'
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    config = {**sample_config, "retry_attempts": 2}
    result = extract_conversation(sample_parsed_conversation, config)
    
    # Should have retried
    assert call_count > 1
    # Should eventually succeed
    assert "project_ideas" in result


def test_extract_conversation_openai_uses_system_prompt(sample_parsed_conversation, sample_config, monkeypatch):
    """Test that OpenAI provider passes system prompt separately."""
    captured_system_prompt = None
    
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        nonlocal captured_system_prompt
        captured_system_prompt = system_prompt
        return '{"project_ideas": [], "problems": [], "workflows": [], "tools_explored": [], "underlying_questions": [], "emotional_signals": {"tone": "neutral", "notes": ""}}'
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    config = {**sample_config, "api_provider": "openai"}
    extract_conversation(sample_parsed_conversation, config)
    
    # OpenAI should have system prompt passed
    assert captured_system_prompt is not None


def test_extract_conversation_anthropic_inline_system(sample_parsed_conversation, sample_config, monkeypatch):
    """Test that Anthropic provider includes system prompt in messages."""
    captured_messages = None
    
    def mock_call_llm(messages, config, schema=None, max_tokens=2048, system_prompt=None):
        nonlocal captured_messages
        captured_messages = messages
        return '{"project_ideas": [], "problems": [], "workflows": [], "tools_explored": [], "underlying_questions": [], "emotional_signals": {"tone": "neutral", "notes": ""}}'
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_call_llm)
    
    config = {**sample_config, "api_provider": "anthropic"}
    extract_conversation(sample_parsed_conversation, config)
    
    # Anthropic should have prompt inline in first message
    assert captured_messages is not None
    assert len(captured_messages) > 0
    # System prompt should be embedded in user content
    assert "Title:" in captured_messages[0]["content"]
