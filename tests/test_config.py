"""
Tests for config.py - Configuration management.
"""
import json
import os
from pathlib import Path

import pytest

from config import (
    DEFAULT_CONFIG,
    get_api_key,
    load_config,
    save_config,
    validate_config
)


def test_default_config_structure():
    """Test that DEFAULT_CONFIG has expected keys."""
    assert "api_provider" in DEFAULT_CONFIG
    assert "model" in DEFAULT_CONFIG
    assert "api_key" in DEFAULT_CONFIG
    assert "requests_per_minute" in DEFAULT_CONFIG
    assert "retry_attempts" in DEFAULT_CONFIG


def test_save_and_load_config(temp_dir, sample_config, monkeypatch):
    """Test saving and loading configuration."""
    config_file = temp_dir / "config.json"
    
    # Patch CONFIG_FILE to use temp directory
    import config as config_module
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    
    # Save config
    save_config(sample_config)
    assert config_file.exists()
    
    # Load config
    loaded = load_config()
    assert loaded["api_provider"] == sample_config["api_provider"]
    assert loaded["model"] == sample_config["model"]
    assert loaded["api_key"] == sample_config["api_key"]


def test_load_config_creates_default(temp_dir, monkeypatch):
    """Test that load_config creates default config if missing."""
    config_file = temp_dir / "config.json"
    
    import config as config_module
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    
    assert not config_file.exists()
    
    config = load_config()
    
    # Should create file
    assert config_file.exists()
    
    # Should have default values
    assert config["api_provider"] == DEFAULT_CONFIG["api_provider"]


def test_get_api_key_from_config(sample_config):
    """Test getting API key from config dict."""
    key = get_api_key(sample_config)
    assert key == "test-key-12345"


def test_get_api_key_from_env_anthropic(sample_config, monkeypatch):
    """Test getting Anthropic API key from environment variable."""
    config = {**sample_config, "api_key": "", "api_provider": "anthropic"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-anthropic")
    
    key = get_api_key(config)
    assert key == "env-key-anthropic"


def test_get_api_key_from_env_openai(sample_config, monkeypatch):
    """Test getting OpenAI API key from environment variable."""
    config = {**sample_config, "api_key": "", "api_provider": "openai"}
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-openai")
    
    key = get_api_key(config)
    assert key == "env-key-openai"


def test_get_api_key_prefers_config_over_env(sample_config, monkeypatch):
    """Test that config api_key takes precedence over environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    
    key = get_api_key(sample_config)
    assert key == "test-key-12345"  # From config, not env


def test_validate_config_valid_openai(sample_config):
    """Test validation passes for valid OpenAI config."""
    is_valid, error = validate_config(sample_config)
    assert is_valid
    assert error == ""


def test_validate_config_valid_anthropic(sample_config):
    """Test validation passes for valid Anthropic config."""
    config = {**sample_config, "api_provider": "anthropic", "model": "claude-sonnet-4"}
    is_valid, error = validate_config(config)
    assert is_valid


def test_validate_config_invalid_provider(sample_config):
    """Test validation fails for invalid provider."""
    config = {**sample_config, "api_provider": "invalid_provider"}
    is_valid, error = validate_config(config)
    assert not is_valid
    assert "Invalid api_provider" in error


def test_validate_config_missing_api_key(sample_config, monkeypatch):
    """Test validation fails when API key is missing."""
    config = {**sample_config, "api_key": ""}
    
    # Clear environment variables
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    
    is_valid, error = validate_config(config)
    assert not is_valid
    assert "No API key" in error


def test_validate_config_missing_model(sample_config):
    """Test validation fails when model is missing."""
    config = {**sample_config, "model": ""}
    is_valid, error = validate_config(config)
    assert not is_valid
    assert "No model specified" in error


def test_validate_config_ollama_no_model(sample_config):
    """Test validation fails for Ollama without model selected."""
    config = {
        **sample_config,
        "api_provider": "ollama",
        "ollama_model": ""
    }
    is_valid, error = validate_config(config)
    assert not is_valid
    assert "No Ollama model selected" in error


def test_config_merge_with_defaults(temp_dir, monkeypatch):
    """Test that loading config merges with defaults for new keys."""
    config_file = temp_dir / "config.json"
    
    import config as config_module
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    
    # Save minimal config
    minimal = {"api_provider": "openai", "model": "gpt-4"}
    save_config(minimal)
    
    # Load should merge with defaults
    loaded = load_config()
    assert loaded["api_provider"] == "openai"
    assert loaded["model"] == "gpt-4"
    assert "requests_per_minute" in loaded  # From defaults
    assert "retry_attempts" in loaded  # From defaults
