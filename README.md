# Resurface

A tool for extracting insights from your ChatGPT conversation history using LLM analysis.

## Overview

Resurface parses ChatGPT export files, extracts meaningful patterns using LLMs (Anthropic Claude or OpenAI GPT), and presents consolidated insights through a web interface.

## Features

- **Parse** ChatGPT JSON exports into clean, linearized conversation files
- **Extract** insights including:
  - Project ideas with motivation and detail level
  - Problems and pain points
  - Workflows being explored or built
  - Tools, APIs, and libraries evaluated
  - Underlying questions and themes
  - Emotional signals (excited, frustrated, curious, stuck)
- **Consolidate** extractions into unified views
- **Visualize** through a Gradio web UI with customizable theming

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install gradio pandas plotly anthropic openai
```

3. Configure your API key:

Either set an environment variable:
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

Or edit `config.json` after first run.

## Usage

### 1. Export your ChatGPT data

Go to ChatGPT Settings > Data Controls > Export data. Place `conversations.json` in `data/`.

### 2. Parse conversations

```bash
python parser.py
```

This creates individual JSON files in `data/parsed/` for conversations with 4+ turns.

### 3. Run the UI

```bash
python ui.py
```

Open the displayed URL to access the web interface where you can:
- Run extractions on parsed conversations
- View and explore extracted insights
- Consolidate findings across conversations
- Configure settings

## Configuration

Edit `config.json`:

```json
{
  "api_provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "",
  "requests_per_minute": 60,
  "retry_attempts": 2,
  "theme_color": "#00ff00"
}
```

| Option | Description |
|--------|-------------|
| `api_provider` | `"anthropic"` or `"openai"` |
| `model` | Model identifier (e.g., `gpt-4o-mini`, `claude-sonnet-4-20250514`) |
| `api_key` | API key (or use environment variable) |
| `requests_per_minute` | Rate limiting for API calls |
| `theme_color` | UI accent color in hex (default: Matrix green `#00ff00`) |

## File Structure

```
Resurface/
├── parser.py          # Parse ChatGPT exports
├── extractor.py       # LLM extraction logic
├── ui.py              # Gradio web interface
├── config.py          # Configuration management
├── config.json        # User configuration
├── assets/            # UI assets (fonts)
└── data/
    ├── conversations.json    # ChatGPT export (you provide)
    ├── parsed/               # Individual conversation files
    ├── extractions/          # LLM extraction results
    └── consolidated/         # Consolidated insights
```

## License

MIT
