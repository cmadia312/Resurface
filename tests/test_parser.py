"""
Tests for parser.py - Conversation parsing from ChatGPT and Claude exports.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from parser import (
    timestamp_to_iso,
    linearize_conversation,
    extract_text_from_parts,
    extract_message,
    process_conversation,
    process_claude_conversation
)


def test_timestamp_to_iso_valid():
    """Test timestamp formatting with valid Unix timestamp."""
    timestamp = 1703275200.0  # 2023-12-22 12:00:00 UTC
    result = timestamp_to_iso(timestamp)
    assert result is not None
    assert result.startswith("2023-12-22")
    assert "T" in result
    assert result.endswith("+00:00") or result.endswith("Z")


def test_timestamp_to_iso_none():
    """Test timestamp formatting with None."""
    result = timestamp_to_iso(None)
    assert result is None


def test_timestamp_to_iso_zero():
    """Test timestamp formatting with zero."""
    result = timestamp_to_iso(0)
    assert result is not None


def test_linearize_conversation_simple():
    """Test linearizing a simple conversation tree."""
    mapping = {
        "msg1": {"id": "msg1", "parent": None},
        "msg2": {"id": "msg2", "parent": "msg1"},
        "msg3": {"id": "msg3", "parent": "msg2"}
    }
    
    path = linearize_conversation(mapping, "msg3")
    
    assert path == ["msg1", "msg2", "msg3"]


def test_linearize_conversation_branching():
    """Test linearizing a branched conversation (takes main path)."""
    mapping = {
        "msg1": {"id": "msg1", "parent": None},
        "msg2": {"id": "msg2", "parent": "msg1"},
        "msg3a": {"id": "msg3a", "parent": "msg2"},
        "msg3b": {"id": "msg3b", "parent": "msg2"}
    }
    
    # Following msg3a path
    path = linearize_conversation(mapping, "msg3a")
    assert path == ["msg1", "msg2", "msg3a"]


def test_linearize_conversation_missing_node():
    """Test linearizing when a node is missing."""
    mapping = {
        "msg1": {"id": "msg1", "parent": None},
        "msg3": {"id": "msg3", "parent": "msg2"}  # msg2 is missing
    }
    
    path = linearize_conversation(mapping, "msg3")
    
    # Should stop at missing node
    assert len(path) >= 1


def test_extract_text_from_parts_simple():
    """Test extracting text from simple string parts."""
    parts = ["Hello", "World"]
    text, has_image = extract_text_from_parts(parts)
    
    assert text == "Hello\nWorld"
    assert has_image is False


def test_extract_text_from_parts_with_image():
    """Test extracting text when parts include image."""
    parts = [
        "Check this out:",
        {"content_type": "image_asset_pointer", "asset_id": "123"}
    ]
    text, has_image = extract_text_from_parts(parts)
    
    assert "Check this out" in text
    assert has_image is True


def test_extract_text_from_parts_dict_with_text():
    """Test extracting text from dict parts with text key."""
    parts = [
        {"text": "Hello"},
        {"text": "World"}
    ]
    text, has_image = extract_text_from_parts(parts)
    
    assert "Hello" in text
    assert "World" in text


def test_extract_message_valid():
    """Test extracting a valid message node."""
    node = {
        "message": {
            "id": "msg1",
            "author": {"role": "user"},
            "create_time": 1703275200.0,
            "content": {
                "content_type": "text",
                "parts": ["Hello world"]
            }
        }
    }
    
    msg = extract_message(node)
    
    assert msg is not None
    assert msg["role"] == "user"
    assert msg["content"] == "Hello world"
    # Timestamp is added later in process_conversation, not in extract_message


def test_extract_message_system_filtered():
    """Test that system messages return None."""
    node = {
        "message": {
            "id": "msg1",
            "author": {"role": "system"},
            "content": {
                "content_type": "text",
                "parts": ["System message"]
            }
        }
    }
    
    msg = extract_message(node)
    assert msg is None


def test_extract_message_empty_content():
    """Test extracting message with empty content."""
    node = {
        "message": {
            "id": "msg1",
            "author": {"role": "user"},
            "content": {
                "content_type": "text",
                "parts": []
            }
        }
    }
    
    msg = extract_message(node)
    # Should return None or empty content depending on implementation
    assert msg is None or msg["content"] == ""


def test_process_conversation_basic(sample_chatgpt_conversation):
    """Test processing a basic ChatGPT conversation."""
    result, error = process_conversation(sample_chatgpt_conversation)
    
    assert error is None
    assert result is not None
    assert "title" in result
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_process_conversation_filters_system(sample_chatgpt_conversation):
    """Test that system messages are filtered out."""
    result, error = process_conversation(sample_chatgpt_conversation)
    
    if result:
        roles = [msg["role"] for msg in result["messages"]]
        assert "system" not in roles


def test_process_conversation_malformed():
    """Test processing malformed conversation."""
    malformed = {
        "title": "Test",
        "create_time": None,
        "mapping": None
    }
    
    result, error = process_conversation(malformed)
    
    # Should handle gracefully
    assert error is not None or result is None or len(result.get("messages", [])) == 0


def test_process_claude_conversation_basic(sample_claude_conversation):
    """Test processing a basic Claude conversation."""
    result, error = process_claude_conversation(sample_claude_conversation)
    
    assert error is None
    assert result is not None
    assert result["title"] == "Project Planning"
    assert result["model"] == "claude"  # Uses 'model' not 'platform'
    assert len(result["messages"]) == 4


def test_process_claude_maps_roles(sample_claude_conversation):
    """Test that Claude sender roles are mapped correctly."""
    result, error = process_claude_conversation(sample_claude_conversation)
    
    if result:
        roles = [msg["role"] for msg in result["messages"]]
        
        # Should only have user and assistant
        assert all(role in ["user", "assistant"] for role in roles)
        # First message should be user
        assert roles[0] == "user"


def test_process_claude_empty_messages():
    """Test processing Claude conversation with no messages."""
    empty_conv = {
        "uuid": "test-123",
        "name": "Empty",
        "created_at": "2023-12-22T12:00:00.000000Z",
        "updated_at": "2023-12-22T12:00:00.000000Z",
        "chat_messages": []
    }
    
    result, error = process_claude_conversation(empty_conv)
    
    # Should handle gracefully
    if result:
        assert len(result["messages"]) == 0


def test_process_conversation_preserves_order(sample_chatgpt_conversation):
    """Test that messages are in chronological order."""
    result, error = process_conversation(sample_chatgpt_conversation)
    
    if result and len(result["messages"]) > 1:
        timestamps = [msg.get("timestamp") for msg in result["messages"]]
        # Filter out None timestamps
        valid_timestamps = [t for t in timestamps if t is not None]
        if valid_timestamps:
            assert valid_timestamps == sorted(valid_timestamps)


def test_extract_message_with_multimodal():
    """Test extracting message with multimodal content."""
    node = {
        "message": {
            "id": "msg1",
            "author": {"role": "user"},
            "create_time": 1703275200.0,
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    "Here's an image:",
                    {"content_type": "image_asset_pointer", "asset_id": "abc"}
                ]
            }
        }
    }
    
    msg = extract_message(node)
    
    if msg:
        # Should extract text and mark image presence
        assert "Here's an image" in msg["content"]
