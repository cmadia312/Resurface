# Resurface

A tool for extracting actionable insights from your ChatGPT and Claude conversation history using LLM analysis.

## Overview

Resurface parses conversation exports from ChatGPT and Claude.ai, extracts meaningful patterns using LLMs (Anthropic Claude, OpenAI GPT, or local models via Ollama), and presents consolidated insights through a web interface. It helps you discover recurring themes, project ideas, problems you're trying to solve, and emotional patterns across your conversations.

## Features

### Core Analysis Pipeline
- **Parse** ChatGPT and Claude JSON exports into clean, normalized conversation files (with configurable minimum turn threshold to filter trivial conversations)
- **Extract** insights using LLM analysis:
  - Project ideas with motivation and detail level
  - Problems and pain points (implicit project seeds)
  - Workflows being explored or built
  - Tools, APIs, and libraries evaluated
  - Underlying questions and recurring themes
  - Emotional signals (excited, frustrated, curious, stuck)
- **Consolidate** extractions by grouping similar items across conversations
- **Categorize** ideas with multi-dimensional scoring:
  - Recurrence (how often it comes up)
  - Passion (emotional investment)
  - Effort (estimated implementation time: 1-5 scale)
  - Monetization potential (1-5 scale)
  - Personal utility (1-5 scale)
- **Synthesize** new project ideas using four strategies:
  - Passion intersections (combining top themes)
  - Problem-solution matching
  - Profile-based generation
  - Time capsule (resurfacing old ideas with new context)

### Web Interface
- **Dashboard**: Overview of processing progress and statistics
- **Extraction**: Run LLM analysis on parsed conversations with progress tracking
- **Browser**: Browse individual conversations and their extractions
- **Ideas**: View all extracted project ideas with filtering
- **Problems**: Explore identified problems and pain points
- **Tools**: See which tools and APIs you're exploring
- **Emotions**: View emotional tone across conversations
- **Consolidated**: View merged insights with source tracking
- **Categories**: See ideas organized by type (Quick Win, Validate, Revive, Someday)
- **Synthesis**: Generate and evaluate AI-created project ideas
- **Settings**: Configure API provider, model, theme, and rate limiting
- **Upload**: Import conversation exports from ChatGPT or Claude (with separate upload tabs for each)
- **Export**: Export all insights to an Obsidian vault

### Additional Capabilities
- **Obsidian vault export**: Export all insights to an Obsidian-compatible markdown vault with wiki-links, YAML frontmatter, and organized folder structure (Ideas, Problems, Tools, Workflows, Conversations)
- **Customizable prompts**: Define custom extraction, consolidation, categorization, and synthesis prompts to tailor analysis to your needs
- **Multiple LLM providers**: Use Anthropic Claude, OpenAI GPT, or run locally with Ollama for privacy
- Resumable extraction with progress tracking
- Configurable rate limiting for API calls
- Customizable UI theming
- Dyslexia-friendly font option (OpenDyslexic)

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the UI

```bash
python ui.py
```

Open the displayed URL to access the web interface. From there you can:
- Upload your conversation exports:
  - **ChatGPT**: Settings > Data Controls > Export data
  - **Claude**: Profile icon > Settings > Account > Export Data
- Configure your API key (OpenAI or Anthropic)
- Parse, extract, and analyze your conversations from both platforms

## Processing Workflow

The full analysis pipeline:

```
Parse → Extract → Consolidate → Categorize → Synthesize
```

1. **Parse**: Converts raw ChatGPT/Claude exports to individual conversation files
2. **Extract**: LLM analyzes each conversation for insights
3. **Consolidate**: Groups similar items across all conversations
4. **Categorize**: Scores and categorizes project ideas
5. **Synthesize**: Generates new ideas from identified patterns

Each step can be run from the web UI, and progress is tracked so you can resume interrupted operations.

## Configuration

Edit `config.json`:

```json
{
  "api_provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "",
  "requests_per_minute": 60,
  "retry_attempts": 2,
  "theme_color": "#00ff00",
  "minimum_turn_threshold": 4,
  "ollama_host": "http://localhost:11434"
}
```

| Option | Description |
|--------|-------------|
| `api_provider` | `"anthropic"`, `"openai"`, or `"ollama"` |
| `model` | Model identifier (e.g., `gpt-4o-mini`, `claude-sonnet-4-20250514`) |
| `api_key` | API key (or use environment variable instead) |
| `requests_per_minute` | Rate limiting for API calls |
| `retry_attempts` | Number of retries for failed API calls |
| `theme_color` | UI accent color in hex (default: `#00ff00`) |
| `minimum_turn_threshold` | Minimum conversation turns to include (default: 4) |
| `ollama_host` | Ollama server URL (default: `http://localhost:11434`) |

## File Structure

```
Resurface/
├── parser.py              # Parse ChatGPT/Claude exports into individual files
├── extractor.py           # LLM extraction logic and prompts
├── runner.py              # Batch extraction orchestration
├── consolidate.py         # Merge similar items across conversations
├── categorize.py          # Score and categorize project ideas
├── synthesizer.py         # Generate new ideas from patterns
├── llm_provider.py        # LLM abstraction (OpenAI, Anthropic, Ollama)
├── schemas.py             # Pydantic data models
├── prompts.py             # Prompt templates and customization
├── obsidian_exporter.py   # Obsidian vault export
├── data_management.py     # Status tracking and reset utilities
├── config.py              # Configuration management
├── ui.py                  # Gradio web interface
├── config.json            # User configuration (create from example)
├── config.example.json    # Configuration template
├── requirements.txt       # Python dependencies
├── assets/                # UI assets (fonts)
└── data/
    ├── chatgpt_conversations.json  # ChatGPT export (uploaded via UI)
    ├── claude_conversations.json   # Claude export (uploaded via UI)
    ├── parsed/            # Individual conversation files (from all sources)
    ├── extractions/       # LLM extraction results
    ├── consolidated/      # Merged insights
    ├── synthesized/       # AI-generated ideas and profiles
    └── obsidian-vault/    # Exported Obsidian vault
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE) for details.
