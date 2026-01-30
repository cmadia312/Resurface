"""
Shared pytest fixtures for Resurface tests.
"""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "test-key-12345",
        "requests_per_minute": 60,
        "retry_attempts": 2,
        "theme_color": "#00ff00",
        "min_turn_threshold": 4,
        "ollama_host": "http://localhost:11434",
        "ollama_model": "",
        "ollama_timeout": 300,
        "prompts": {}
    }


@pytest.fixture
def sample_chatgpt_conversation() -> dict:
    """Sample ChatGPT conversation export."""
    return {
        "title": "Test Conversation",
        "create_time": 1703275200.0,
        "update_time": 1703278800.0,
        "mapping": {
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1",
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": ["System message"]},
                    "create_time": 1703275200.0
                },
                "parent": None,
                "children": ["msg2"]
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "id": "msg2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello, I want to build a task manager app"]},
                    "create_time": 1703275260.0
                },
                "parent": "msg1",
                "children": ["msg3"]
            },
            "msg3": {
                "id": "msg3",
                "message": {
                    "id": "msg3",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["That's a great idea! What features are you thinking?"]},
                    "create_time": 1703275320.0
                },
                "parent": "msg2",
                "children": ["msg4"]
            },
            "msg4": {
                "id": "msg4",
                "message": {
                    "id": "msg4",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["I want priorities, due dates, and tags"]},
                    "create_time": 1703275380.0
                },
                "parent": "msg3",
                "children": ["msg5"]
            },
            "msg5": {
                "id": "msg5",
                "message": {
                    "id": "msg5",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Here's how you could structure that..."]},
                    "create_time": 1703275440.0
                },
                "parent": "msg4",
                "children": []
            }
        },
        "current_node": "msg5"
    }


@pytest.fixture
def sample_claude_conversation() -> dict:
    """Sample Claude conversation export."""
    return {
        "uuid": "test-uuid-123",
        "name": "Project Planning",
        "created_at": "2023-12-22T12:00:00.000000Z",
        "updated_at": "2023-12-22T13:00:00.000000Z",
        "chat_messages": [
            {
                "uuid": "msg1",
                "text": "I need help organizing my notes",
                "sender": "human",
                "created_at": "2023-12-22T12:00:00.000000Z"
            },
            {
                "uuid": "msg2",
                "text": "I'd be happy to help! What note-taking system are you using?",
                "sender": "assistant",
                "created_at": "2023-12-22T12:01:00.000000Z"
            },
            {
                "uuid": "msg3",
                "text": "Just random text files. It's a mess.",
                "sender": "human",
                "created_at": "2023-12-22T12:02:00.000000Z"
            },
            {
                "uuid": "msg4",
                "text": "Let's create a structure for you...",
                "sender": "assistant",
                "created_at": "2023-12-22T12:03:00.000000Z"
            }
        ]
    }


@pytest.fixture
def sample_parsed_conversation() -> dict:
    """Sample parsed conversation (output from parser.py)."""
    return {
        "id": "test-conv-123",
        "title": "Building a Task Manager",
        "created": "2023-12-22T12:00:00Z",
        "updated": "2023-12-22T13:00:00Z",
        "platform": "chatgpt",
        "message_count": 4,
        "messages": [
            {
                "role": "user",
                "content": "I want to build a task manager app",
                "timestamp": "2023-12-22T12:00:00Z"
            },
            {
                "role": "assistant",
                "content": "That's a great idea! What features are you thinking?",
                "timestamp": "2023-12-22T12:01:00Z"
            },
            {
                "role": "user",
                "content": "I want priorities, due dates, and tags",
                "timestamp": "2023-12-22T12:02:00Z"
            },
            {
                "role": "assistant",
                "content": "Here's how you could structure that...",
                "timestamp": "2023-12-22T12:03:00Z"
            }
        ]
    }


@pytest.fixture
def sample_extraction_result() -> dict:
    """Sample extraction result (output from extractor.py)."""
    return {
        "project_ideas": [
            {
                "idea": "Task manager app with priorities, due dates, and tags",
                "motivation": "User needs to organize their tasks better",
                "detail_level": "sketched"
            }
        ],
        "problems": [
            {
                "problem": "Current task management is disorganized",
                "context": "User is overwhelmed with tasks"
            }
        ],
        "workflows": [],
        "tools_explored": [],
        "underlying_questions": ["How to structure task data effectively?"],
        "emotional_signals": {
            "tone": "curious",
            "notes": "User seems motivated to build something"
        }
    }


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response in JSON format."""
    return json.dumps({
        "project_ideas": [
            {
                "idea": "AI-powered note organizer",
                "motivation": "Current system is chaotic",
                "detail_level": "vague"
            }
        ],
        "problems": [],
        "workflows": [],
        "tools_explored": ["Python", "SQLite"],
        "underlying_questions": [],
        "emotional_signals": {
            "tone": "frustrated",
            "notes": "User is tired of manual organization"
        }
    })
