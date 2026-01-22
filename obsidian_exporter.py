#!/usr/bin/env python3
"""
Obsidian Vault Exporter for Resurface.

Exports all Resurface data to an Obsidian-compatible markdown vault
with wiki-links, YAML frontmatter, and thematic organization.
"""
import json
import os
import platform
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path("data")
PARSED_DIR = DATA_DIR / "parsed"
EXTRACTIONS_DIR = DATA_DIR / "extractions"
CONSOLIDATED_DIR = DATA_DIR / "consolidated"
SYNTHESIZED_DIR = DATA_DIR / "synthesized"
VAULT_DIR = DATA_DIR / "obsidian-vault"
STATUS_FILE = DATA_DIR / "export_status.json"


# =============================================================================
# STATUS MANAGEMENT
# =============================================================================

def safe_replace(src, dst, retries=3, delay=0.1):
    """Cross-platform atomic file replace with Windows retry logic."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if platform.system() == 'Windows' and attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def update_status(message: str, progress_pct: float = None,
                  complete: bool = False, error: bool = False):
    """Atomically update export status for UI polling."""
    status = {
        "message": message,
        "progress": progress_pct,
        "complete": complete,
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid()
    }
    STATUS_FILE.parent.mkdir(exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=STATUS_FILE.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
        safe_replace(tmp_path, STATUS_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_status() -> dict:
    """Get current export status."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {"message": "No export in progress", "complete": False}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def slugify(text: str) -> str:
    """Convert text to URL-safe slug for filenames."""
    if not text:
        return "untitled"
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = text.strip('-')
    return text[:50] if text else "untitled"


def sanitize_yaml_string(value: str) -> str:
    """Escape YAML special characters in string values."""
    if not isinstance(value, str):
        return str(value)
    # If contains special chars, wrap in quotes
    if any(c in value for c in [':', '#', '[', ']', '{', '}', '"', "'", '\n', '|', '>']):
        # Escape existing double quotes and wrap
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    return value


def format_frontmatter(metadata: dict) -> str:
    """Generate YAML frontmatter block."""
    lines = ["---"]

    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in value:
                    if isinstance(item, str):
                        lines.append(f"  - {sanitize_yaml_string(item)}")
                    else:
                        lines.append(f"  - {item}")
        elif isinstance(value, dict):
            if not value:
                lines.append(f"{key}: {{}}")
            else:
                lines.append(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, str):
                        lines.append(f"  {k}: {sanitize_yaml_string(v)}")
                    else:
                        lines.append(f"  {k}: {v}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, str):
            lines.append(f"{key}: {sanitize_yaml_string(value)}")
        else:
            lines.append(f"{key}: {value}")

    lines.append("---")
    return "\n".join(lines)


def generate_tags(item: dict, note_type: str) -> list[str]:
    """Generate hierarchical tags for a note."""
    tags = [f"type/{note_type}"]

    # Add emotion tag
    emotion = item.get("emotion") or item.get("tone")
    if emotion:
        tags.append(f"emotion/{emotion}")

    # Add category tag (for ideas)
    category = item.get("category")
    if category:
        tags.append(f"category/{category}")

    # Add tool tags
    tools = item.get("tools_explored", []) + item.get("tools", [])
    for tool in tools:
        tags.append(f"tool/{slugify(tool)}")

    # Add month tag
    date = item.get("created") or item.get("conversation_date")
    if date:
        month = date[:7] if isinstance(date, str) else date.strftime("%Y-%m")
        tags.append(f"month/{month}")

    # Add detail level tag
    detail = item.get("detail_level")
    if detail:
        tags.append(f"detail/{detail}")

    return tags


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_all_parsed() -> list[dict]:
    """Load all parsed conversations."""
    conversations = []
    if not PARSED_DIR.exists():
        return conversations

    for file_path in PARSED_DIR.glob("*.json"):
        if file_path.name.startswith("_"):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return sorted(conversations, key=lambda x: x.get("created", ""), reverse=True)


def load_all_extractions() -> dict[str, dict]:
    """Load all extractions, keyed by conversation_id."""
    extractions = {}
    if not EXTRACTIONS_DIR.exists():
        return extractions

    for file_path in EXTRACTIONS_DIR.glob("*.json"):
        if file_path.name.startswith("_"):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conv_id = data.get("conversation_id")
                if conv_id:
                    extractions[conv_id] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return extractions


def load_consolidated() -> dict:
    """Load consolidated data."""
    path = CONSOLIDATED_DIR / "consolidated.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_categorized() -> dict:
    """Load categorized ideas."""
    path = CONSOLIDATED_DIR / "categorized.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_passion_profile() -> dict:
    """Load passion profile from synthesis."""
    path = SYNTHESIZED_DIR / "passion_profile.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_generated_ideas() -> list[dict]:
    """Load generated ideas from synthesis."""
    path = SYNTHESIZED_DIR / "generated_ideas.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("ideas", []) if isinstance(data, dict) else data
    return []


def load_saved_ideas() -> list[dict]:
    """Load user-saved ideas."""
    path = SYNTHESIZED_DIR / "saved_ideas.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


# =============================================================================
# LINK REGISTRY
# =============================================================================

class LinkRegistry:
    """
    Tracks all notes and their relationships for link generation.

    This allows us to:
    1. Generate correct wiki-links between notes
    2. Build tool/theme hub pages with backlinks
    3. Track co-occurrence for "related" sections
    """

    def __init__(self):
        self.conversations = {}  # id -> {path, title, tools, ideas, ...}
        self.ideas = {}          # name -> {path, sources, tools, category}
        self.problems = {}       # name -> {path, sources, tools}
        self.workflows = {}      # name -> {path, sources}
        self.tools = defaultdict(lambda: {
            "conversations": [],
            "ideas": [],
            "problems": [],
            "workflows": []
        })
        self.themes = {}         # theme -> {strength, evidence, ideas}
        self.tool_cooccurrence = defaultdict(lambda: defaultdict(int))

    def register_conversation(self, conv_id: str, data: dict):
        """Register a conversation for linking."""
        created = data.get("created", "0000-00-00")
        month = created[:7] if created else "0000-00"
        title = data.get("title", "Untitled")
        slug = f"{slugify(title)}-{conv_id[:8]}"
        path = f"Conversations/{month}/{slug}"

        self.conversations[conv_id] = {
            "path": path,
            "title": title,
            "tools": data.get("tools_explored", []),
            "ideas": [i.get("idea", "") for i in data.get("project_ideas", [])],
            "problems": [p.get("problem", "") for p in data.get("problems", [])],
            "workflows": [w.get("workflow", "") for w in data.get("workflows", [])],
            "emotion": data.get("emotion", "neutral"),
            "date": created[:10] if created else ""
        }

        # Register tool relationships
        tools = data.get("tools_explored", [])
        for tool in tools:
            self.tools[tool]["conversations"].append(conv_id)

        # Track co-occurrence for related tools
        for i, t1 in enumerate(tools):
            for t2 in tools[i+1:]:
                self.tool_cooccurrence[t1][t2] += 1
                self.tool_cooccurrence[t2][t1] += 1

    def register_idea(self, idea: dict, category: str = "someday"):
        """Register an idea for linking."""
        name = idea.get("name", "")
        if not name:
            return

        slug = slugify(name)
        category_folders = {
            "quick_win": "Quick Wins",
            "validate": "Validate",
            "revive": "Revive",
            "someday": "Someday"
        }
        category_folder = category_folders.get(category, "Someday")
        path = f"Ideas/{category_folder}/{slug}"

        source_ids = idea.get("source_ids", [])
        tools = self._get_tools_from_sources(source_ids)

        self.ideas[name] = {
            "path": path,
            "sources": source_ids,
            "tools": tools,
            "category": category
        }

        # Register with tools
        for tool in tools:
            if name not in self.tools[tool]["ideas"]:
                self.tools[tool]["ideas"].append(name)

    def register_problem(self, problem: dict):
        """Register a problem for linking."""
        name = problem.get("name", "")
        if not name:
            return

        slug = slugify(name)
        path = f"Problems/{slug}"

        source_ids = problem.get("source_ids", [])
        tools = self._get_tools_from_sources(source_ids)

        self.problems[name] = {
            "path": path,
            "sources": source_ids,
            "tools": tools
        }

        for tool in tools:
            if name not in self.tools[tool]["problems"]:
                self.tools[tool]["problems"].append(name)

    def register_workflow(self, workflow: dict):
        """Register a workflow for linking."""
        name = workflow.get("name", "")
        if not name:
            return

        slug = slugify(name)
        path = f"Workflows/{slug}"

        source_ids = workflow.get("source_ids", [])

        self.workflows[name] = {
            "path": path,
            "sources": source_ids
        }

        # Register with tools from sources
        tools = self._get_tools_from_sources(source_ids)
        for tool in tools:
            if name not in self.tools[tool]["workflows"]:
                self.tools[tool]["workflows"].append(name)

    def register_theme(self, theme: dict):
        """Register a theme from passion profile."""
        name = theme.get("theme", "")
        if not name:
            return

        self.themes[name] = {
            "strength": theme.get("strength", 3),
            "evidence": theme.get("evidence", ""),
            "ideas": [],
            "conversations": []
        }

    def _get_tools_from_sources(self, source_ids: list) -> list:
        """Get all tools from source conversations."""
        tools = set()
        for src_id in source_ids:
            if src_id in self.conversations:
                tools.update(self.conversations[src_id].get("tools", []))
        return list(tools)

    def get_related_tools(self, tool: str, limit: int = 5) -> list[str]:
        """Get tools frequently used together with the given tool."""
        cooccurrences = self.tool_cooccurrence.get(tool, {})
        sorted_tools = sorted(cooccurrences.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tools[:limit]]

    def make_link(self, target_type: str, identifier: str, display_text: str = None) -> str:
        """
        Generate wiki-link to a registered note.

        Args:
            target_type: One of 'conversation', 'idea', 'problem', 'workflow', 'tool', 'theme'
            identifier: The ID or name of the target
            display_text: Optional custom display text

        Returns:
            Wiki-link string like [[path|display]] or plain text if not found
        """
        path = None
        default_display = identifier

        if target_type == "conversation":
            if identifier in self.conversations:
                path = self.conversations[identifier]["path"]
                default_display = self.conversations[identifier]["title"]
        elif target_type == "idea":
            if identifier in self.ideas:
                path = self.ideas[identifier]["path"]
                default_display = identifier
        elif target_type == "problem":
            if identifier in self.problems:
                path = self.problems[identifier]["path"]
                default_display = identifier
        elif target_type == "workflow":
            if identifier in self.workflows:
                path = self.workflows[identifier]["path"]
                default_display = identifier
        elif target_type == "tool":
            path = f"Tools/{slugify(identifier)}"
            default_display = identifier
        elif target_type == "theme":
            path = f"Themes/{slugify(identifier)}"
            default_display = identifier

        if path:
            display = display_text or default_display
            return f"[[{path}|{display}]]"

        return display_text or identifier


# =============================================================================
# NOTE GENERATORS
# =============================================================================

def generate_conversation_note(conv: dict, extraction: dict, registry: LinkRegistry) -> str:
    """Generate markdown for a conversation note."""
    conv_id = conv.get("id", "")
    title = conv.get("title", "Untitled")

    ext = extraction.get("extraction", {}) if extraction else {}
    emotion = ext.get("emotional_signals", {})

    # Build frontmatter
    frontmatter = {
        "id": conv_id,
        "title": title,
        "created": conv.get("created", "")[:10] if conv.get("created") else "",
        "updated": conv.get("updated", "")[:10] if conv.get("updated") else "",
        "model": conv.get("model", "unknown"),
        "message_count": conv.get("message_count", 0),
        "turn_count": conv.get("turn_count", 0),
        "emotion": emotion.get("tone", "neutral"),
        "emotion_notes": emotion.get("notes", ""),
        "tools": ext.get("tools_explored", []),
        "ideas_extracted": [i.get("idea", "")[:50] for i in ext.get("project_ideas", [])],
        "problems_extracted": [p.get("problem", "")[:50] for p in ext.get("problems", [])],
        "type": "conversation",
        "tags": generate_tags({
            "emotion": emotion.get("tone", "neutral"),
            "tools_explored": ext.get("tools_explored", []),
            "created": conv.get("created", "")
        }, "conversation")
    }

    content = [format_frontmatter(frontmatter), "", f"# {title}", ""]

    # Metadata callout
    content.append("> [!info] Metadata")
    content.append(f"> **Date:** {conv.get('created', '')[:10]}  ")
    content.append(f"> **Model:** {conv.get('model', 'unknown')}  ")
    if emotion.get("tone"):
        content.append(f"> **Emotion:** {emotion.get('tone')}  ")
    if emotion.get("notes"):
        content.append(f"> **Notes:** {emotion.get('notes')}  ")
    content.append("")

    # Tools section
    tools = ext.get("tools_explored", [])
    if tools:
        content.append("## Tools Explored")
        for tool in tools:
            content.append(f"- {registry.make_link('tool', tool)}")
        content.append("")

    # Ideas section
    ideas = ext.get("project_ideas", [])
    if ideas:
        content.append("## Ideas Mentioned")
        for idea in ideas:
            idea_name = idea.get("idea", "")
            # Try to link to consolidated idea if it exists
            content.append(f"- {registry.make_link('idea', idea_name)}")
            if idea.get("motivation"):
                content.append(f"  - *Motivation:* {idea.get('motivation')}")
        content.append("")

    # Problems section
    problems = ext.get("problems", [])
    if problems:
        content.append("## Problems Discussed")
        for prob in problems:
            prob_name = prob.get("problem", "")
            content.append(f"- {registry.make_link('problem', prob_name)}")
            if prob.get("context"):
                content.append(f"  - *Context:* {prob.get('context')}")
        content.append("")

    # Workflows section
    workflows = ext.get("workflows", [])
    if workflows:
        content.append("## Workflows")
        for wf in workflows:
            wf_name = wf.get("workflow", "")
            status = wf.get("status", "")
            content.append(f"- {registry.make_link('workflow', wf_name)} ({status})")
        content.append("")

    # Underlying questions
    questions = ext.get("underlying_questions", [])
    if questions:
        content.append("## Underlying Questions")
        for q in questions:
            content.append(f"- {q}")
        content.append("")

    # Full conversation
    content.append("## Conversation")
    content.append("")
    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown").title()
        text = msg.get("content", "")
        content.append(f"### {role}")
        content.append(text)
        content.append("")

    return "\n".join(content)


def generate_idea_note(idea: dict, registry: LinkRegistry) -> str:
    """Generate markdown for a consolidated idea note."""
    name = idea.get("name", "Untitled Idea")
    category = idea.get("category", "someday")
    scores = idea.get("scores", {})

    # Get tools from registry
    idea_data = registry.ideas.get(name, {})
    tools = idea_data.get("tools", [])

    frontmatter = {
        "id": f"idea-{slugify(name)}",
        "name": name,
        "description": idea.get("description", ""),
        "category": category,
        "occurrences": idea.get("occurrences", 1),
        "date_range": idea.get("date_range", []),
        "source_ids": idea.get("source_ids", []),
        "motivations": idea.get("motivations", []),
        "detail_levels": idea.get("detail_levels", []),
        "scores": scores,
        "composite_score": idea.get("composite_score", 0),
        "type": "idea",
        "tags": [
            "type/idea",
            f"category/{category}",
        ] + [f"tool/{slugify(t)}" for t in tools]
    }

    content = [format_frontmatter(frontmatter), "", f"# {name}", ""]

    # Summary callout
    if idea.get("description"):
        content.append("> [!summary]")
        content.append(f"> {idea.get('description')}")
        content.append("")

    # Scores table
    if scores:
        content.append("## Scores")
        content.append("| Dimension | Score |")
        content.append("|-----------|-------|")
        for dim in ["passion", "recurrence", "effort", "monetization", "personal_utility"]:
            if dim in scores:
                content.append(f"| {dim.replace('_', ' ').title()} | {scores[dim]}/5 |")
        if idea.get("composite_score"):
            content.append(f"| **Composite** | **{idea.get('composite_score')}** |")
        content.append("")

    # Evolution
    if idea.get("evolution"):
        content.append("## Evolution")
        content.append(idea["evolution"])
        content.append("")

    # Motivations
    motivations = idea.get("motivations", [])
    if motivations:
        content.append("## Motivations")
        for m in motivations:
            content.append(f"- {m}")
        content.append("")

    # Related tools
    if tools:
        content.append("## Related Tools")
        for tool in tools:
            content.append(f"- {registry.make_link('tool', tool)}")
        content.append("")

    # Source conversations
    sources = idea.get("source_ids", [])
    if sources:
        content.append("## Source Conversations")
        for src_id in sources:
            content.append(f"- {registry.make_link('conversation', src_id)}")
        content.append("")

    return "\n".join(content)


def generate_problem_note(problem: dict, registry: LinkRegistry) -> str:
    """Generate markdown for a problem note."""
    name = problem.get("name", "Unknown Problem")

    # Get tools from registry
    prob_data = registry.problems.get(name, {})
    tools = prob_data.get("tools", [])

    frontmatter = {
        "id": f"problem-{slugify(name)}",
        "name": name,
        "description": problem.get("description", ""),
        "occurrences": problem.get("occurrences", 1),
        "date_range": problem.get("date_range", []),
        "source_ids": problem.get("source_ids", []),
        "type": "problem",
        "tags": ["type/problem"] + [f"tool/{slugify(t)}" for t in tools]
    }

    content = [format_frontmatter(frontmatter), "", f"# {name}", ""]

    # Description
    if problem.get("description"):
        content.append("## Description")
        content.append(problem["description"])
        content.append("")

    # Contexts
    contexts = problem.get("contexts", [])
    if contexts:
        content.append("## Contexts")
        for ctx in contexts:
            content.append(f"- {ctx}")
        content.append("")

    # Related tools
    if tools:
        content.append("## Related Tools")
        for tool in tools:
            content.append(f"- {registry.make_link('tool', tool)}")
        content.append("")

    # Source conversations
    sources = problem.get("source_ids", [])
    if sources:
        content.append("## Source Conversations")
        for src_id in sources:
            content.append(f"- {registry.make_link('conversation', src_id)}")
        content.append("")

    return "\n".join(content)


def generate_workflow_note(workflow: dict, registry: LinkRegistry) -> str:
    """Generate markdown for a workflow note."""
    name = workflow.get("name", "Unknown Workflow")

    frontmatter = {
        "id": f"workflow-{slugify(name)}",
        "name": name,
        "description": workflow.get("description", ""),
        "occurrences": workflow.get("occurrences", 1),
        "date_range": workflow.get("date_range", []),
        "source_ids": workflow.get("source_ids", []),
        "statuses": workflow.get("statuses", []),
        "type": "workflow",
        "tags": ["type/workflow"]
    }

    content = [format_frontmatter(frontmatter), "", f"# {name}", ""]

    # Description
    if workflow.get("description"):
        content.append("## Description")
        content.append(workflow["description"])
        content.append("")

    # Status progression
    statuses = workflow.get("statuses", [])
    if statuses:
        content.append("## Status History")
        for status in statuses:
            content.append(f"- {status}")
        content.append("")

    # Source conversations
    sources = workflow.get("source_ids", [])
    if sources:
        content.append("## Source Conversations")
        for src_id in sources:
            content.append(f"- {registry.make_link('conversation', src_id)}")
        content.append("")

    return "\n".join(content)


def generate_tool_note(tool: str, registry: LinkRegistry) -> str:
    """Generate markdown for a tool hub note."""
    tool_data = registry.tools.get(tool, {})
    conversations = tool_data.get("conversations", [])
    ideas = tool_data.get("ideas", [])
    problems = tool_data.get("problems", [])
    workflows = tool_data.get("workflows", [])
    related = registry.get_related_tools(tool, 5)

    frontmatter = {
        "tool": tool,
        "mentions": len(conversations),
        "type": "tool",
        "tags": ["type/tool", f"tool/{slugify(tool)}"]
    }

    content = [format_frontmatter(frontmatter), "", f"# {tool}", ""]

    content.append(f"> [!info] Mentioned in **{len(conversations)}** conversations")
    content.append("")

    # Ideas using this tool
    if ideas:
        content.append(f"## Ideas Using {tool}")
        for idea in ideas:
            content.append(f"- {registry.make_link('idea', idea)}")
        content.append("")

    # Problems related to this tool
    if problems:
        content.append("## Related Problems")
        for prob in problems:
            content.append(f"- {registry.make_link('problem', prob)}")
        content.append("")

    # Workflows using this tool
    if workflows:
        content.append("## Workflows")
        for wf in workflows:
            content.append(f"- {registry.make_link('workflow', wf)}")
        content.append("")

    # Conversations
    if conversations:
        content.append(f"## Conversations Mentioning {tool}")
        # Sort by date (most recent first) and limit
        sorted_convs = sorted(
            conversations,
            key=lambda c: registry.conversations.get(c, {}).get("date", ""),
            reverse=True
        )
        for conv_id in sorted_convs[:20]:
            content.append(f"- {registry.make_link('conversation', conv_id)}")
        if len(conversations) > 20:
            content.append(f"- *...and {len(conversations) - 20} more*")
        content.append("")

    # Related tools
    if related:
        content.append("## Related Tools")
        content.append("*Tools frequently used together with this one:*")
        content.append("")
        for rel_tool in related:
            count = registry.tool_cooccurrence[tool][rel_tool]
            content.append(f"- {registry.make_link('tool', rel_tool)} ({count} co-occurrences)")
        content.append("")

    return "\n".join(content)


def generate_theme_note(theme: dict, registry: LinkRegistry) -> str:
    """Generate markdown for a theme hub note."""
    name = theme.get("theme", "Unknown Theme")
    strength = theme.get("strength", 3)
    evidence = theme.get("evidence", "")

    strength_label = "high" if strength >= 4 else "medium" if strength >= 2 else "low"

    frontmatter = {
        "theme": name,
        "strength": strength,
        "type": "theme",
        "tags": ["type/theme", f"strength/{strength_label}"]
    }

    content = [format_frontmatter(frontmatter), "", f"# {name}", ""]

    content.append(f"> [!abstract] Core Theme (Strength: {strength}/5)")
    if evidence:
        content.append(f"> {evidence}")
    content.append("")

    # Note: We could add related ideas here by matching keywords
    # For now, Obsidian's backlinks will show connections

    content.append("## Related Content")
    content.append("*See backlinks for ideas, problems, and conversations related to this theme.*")
    content.append("")

    return "\n".join(content)


def generate_passion_profile_note(profile: dict, registry: LinkRegistry) -> str:
    """Generate markdown for the passion profile note."""
    frontmatter = {
        "type": "profile",
        "generated_at": profile.get("generated_at", ""),
        "tags": ["type/profile"]
    }

    content = [format_frontmatter(frontmatter), "", "# Passion Profile", ""]

    # Summary
    if profile.get("summary"):
        content.append("> [!abstract] Summary")
        content.append(f"> {profile['summary']}")
        content.append("")

    # Date range
    date_range = profile.get("date_range", {})
    if date_range:
        content.append(f"**Analysis Period:** {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
        content.append("")

    # Core themes
    themes = profile.get("core_themes", [])
    if themes:
        content.append("## Core Themes")
        content.append("| Theme | Strength | Evidence |")
        content.append("|-------|----------|----------|")
        for theme in themes:
            name = theme.get("theme", "")
            strength = theme.get("strength", 3)
            evidence = theme.get("evidence", "")[:60]
            link = registry.make_link("theme", name)
            content.append(f"| {link} | {strength}/5 | {evidence}... |")
        content.append("")

    # Tool expertise
    tools = profile.get("tool_expertise", [])
    if tools:
        content.append("## Tool Expertise")
        for tool in tools:
            name = tool.get("tool", "")
            mentions = tool.get("mentions", 0)
            contexts = tool.get("contexts", [])
            link = registry.make_link("tool", name)
            content.append(f"- {link} ({mentions} mentions)")
            if contexts:
                for ctx in contexts[:2]:
                    content.append(f"  - {ctx}")
        content.append("")

    # Recurring problems
    problems = profile.get("recurring_problems", [])
    if problems:
        content.append("## Recurring Problems")
        for prob in problems:
            name = prob.get("problem", "")
            frequency = prob.get("frequency", 0)
            impact = prob.get("impact", "")
            content.append(f"- **{name}** (frequency: {frequency})")
            if impact:
                content.append(f"  - Impact: {impact}")
        content.append("")

    # Emotional patterns
    emotions = profile.get("emotional_patterns", {})
    if emotions:
        content.append("## Emotional Patterns")
        if emotions.get("excited_about"):
            content.append("**Excited about:**")
            for item in emotions["excited_about"]:
                content.append(f"- {item}")
            content.append("")
        if emotions.get("frustrated_by"):
            content.append("**Frustrated by:**")
            for item in emotions["frustrated_by"]:
                content.append(f"- {item}")
            content.append("")
        if emotions.get("curious_about"):
            content.append("**Curious about:**")
            for item in emotions["curious_about"]:
                content.append(f"- {item}")
            content.append("")

    # Underlying questions
    questions = profile.get("underlying_questions", [])
    if questions:
        content.append("## Underlying Questions")
        for q in questions:
            content.append(f"- {q}")
        content.append("")

    # High passion ideas
    high_passion = profile.get("high_passion_ideas", [])
    if high_passion:
        content.append("## High Passion Ideas")
        for idea in high_passion:
            name = idea.get("name", "")
            score = idea.get("passion_score", 0)
            link = registry.make_link("idea", name)
            content.append(f"- {link} (Passion: {score}/5)")
        content.append("")

    return "\n".join(content)


def generate_generated_idea_note(idea: dict, registry: LinkRegistry) -> str:
    """Generate markdown for an AI-generated idea note."""
    name = idea.get("name", "Untitled Idea")
    strategy = idea.get("strategy", "unknown")
    scores = idea.get("scores", {})

    # Determine subfolder based on strategy
    strategy_folders = {
        "intersection": "Intersection Ideas",
        "problem_solution": "Problem Solutions",
        "profile_based": "Profile Based",
        "time_capsule": "Time Capsule"
    }
    folder = strategy_folders.get(strategy, "Other")

    frontmatter = {
        "id": idea.get("id", f"gen-{slugify(name)}"),
        "name": name,
        "strategy": strategy,
        "description": idea.get("description", ""),
        "scores": scores,
        "composite_score": idea.get("composite_score", 0),
        "status": idea.get("status", "generated"),
        "type": "generated",
        "tags": ["type/generated", f"strategy/{strategy}"]
    }

    content = [format_frontmatter(frontmatter), "", f"# {name}", ""]

    # Description
    if idea.get("description"):
        content.append("> [!idea]")
        content.append(f"> {idea['description']}")
        content.append("")

    # Strategy-specific content
    if strategy == "intersection" and idea.get("themes_combined"):
        content.append("## Themes Combined")
        for theme in idea["themes_combined"]:
            content.append(f"- {registry.make_link('theme', theme)}")
        content.append("")
        if idea.get("why_exciting"):
            content.append("## Why Exciting")
            content.append(idea["why_exciting"])
            content.append("")

    if strategy == "problem_solution" and idea.get("problem_addressed"):
        content.append("## Problem Addressed")
        content.append(idea["problem_addressed"])
        content.append("")
        if idea.get("why_practical"):
            content.append("## Why Practical")
            content.append(idea["why_practical"])
            content.append("")

    if strategy == "profile_based":
        if idea.get("profile_alignment"):
            content.append("## Profile Alignment")
            content.append(idea["profile_alignment"])
            content.append("")
        if idea.get("novelty_factor"):
            content.append("## Novelty Factor")
            content.append(idea["novelty_factor"])
            content.append("")

    if strategy == "time_capsule":
        if idea.get("original_idea"):
            content.append("## Original Idea")
            content.append(idea["original_idea"])
            content.append("")
        if idea.get("letter_from_past"):
            content.append("## Letter from the Past")
            content.append(idea["letter_from_past"])
            content.append("")

    # Tools suggested
    tools = idea.get("tools_suggested", []) or idea.get("tools_used", [])
    if tools:
        content.append("## Suggested Tools")
        for tool in tools:
            content.append(f"- {registry.make_link('tool', tool)}")
        content.append("")

    # Scores
    if scores:
        content.append("## Scores")
        content.append("| Dimension | Score |")
        content.append("|-----------|-------|")
        for dim, score in scores.items():
            content.append(f"| {dim.replace('_', ' ').title()} | {score}/5 |")
        if idea.get("composite_score"):
            content.append(f"| **Composite** | **{idea['composite_score']}** |")
        content.append("")

    return "\n".join(content)


# =============================================================================
# MOC (MAP OF CONTENT) GENERATORS
# =============================================================================

def generate_tools_moc(registry: LinkRegistry) -> str:
    """Generate Tools Map of Content."""
    content = [
        "---",
        "type: moc",
        "tags:",
        "  - type/moc",
        "  - map/tools",
        "---",
        "",
        "# Tools Map of Content",
        ""
    ]

    # Sort by frequency
    sorted_tools = sorted(
        registry.tools.items(),
        key=lambda x: len(x[1].get("conversations", [])),
        reverse=True
    )

    if not sorted_tools:
        content.append("*No tools found. Run extraction first.*")
        return "\n".join(content)

    content.append("## By Frequency")
    content.append("| Tool | Mentions | Ideas | Problems |")
    content.append("|------|----------|-------|----------|")
    for tool, data in sorted_tools[:30]:
        mentions = len(data.get("conversations", []))
        ideas = len(data.get("ideas", []))
        problems = len(data.get("problems", []))
        link = registry.make_link("tool", tool)
        content.append(f"| {link} | {mentions} | {ideas} | {problems} |")

    if len(sorted_tools) > 30:
        content.append(f"\n*...and {len(sorted_tools) - 30} more tools*")

    content.append("")
    return "\n".join(content)


def generate_themes_moc(registry: LinkRegistry, profile: dict) -> str:
    """Generate Themes Map of Content."""
    content = [
        "---",
        "type: moc",
        "tags:",
        "  - type/moc",
        "  - map/themes",
        "---",
        "",
        "# Themes Map of Content",
        ""
    ]

    themes = profile.get("core_themes", [])
    if not themes:
        content.append("*No themes found. Run synthesis first to generate passion profile.*")
        return "\n".join(content)

    # Sort by strength
    sorted_themes = sorted(themes, key=lambda x: x.get("strength", 0), reverse=True)

    content.append("## Core Themes by Strength")
    content.append("")
    for theme in sorted_themes:
        name = theme.get("theme", "")
        strength = theme.get("strength", 3)
        evidence = theme.get("evidence", "")
        link = registry.make_link("theme", name)

        strength_bar = "" * strength + "" * (5 - strength)
        content.append(f"### {link}")
        content.append(f"**Strength:** {strength_bar} ({strength}/5)")
        content.append("")
        if evidence:
            content.append(f"> {evidence}")
            content.append("")

    return "\n".join(content)


def generate_timeline_moc(registry: LinkRegistry) -> str:
    """Generate Timeline Map of Content."""
    content = [
        "---",
        "type: moc",
        "tags:",
        "  - type/moc",
        "  - map/timeline",
        "---",
        "",
        "# Timeline",
        ""
    ]

    # Group conversations by month
    by_month = defaultdict(list)
    for conv_id, data in registry.conversations.items():
        date = data.get("date", "")
        if date:
            month = date[:7]
            by_month[month].append((conv_id, data))

    if not by_month:
        content.append("*No conversations found.*")
        return "\n".join(content)

    # Sort months descending
    for month in sorted(by_month.keys(), reverse=True):
        convs = by_month[month]
        content.append(f"## {month}")
        content.append("")

        # Sort conversations by date within month
        sorted_convs = sorted(convs, key=lambda x: x[1].get("date", ""), reverse=True)

        for conv_id, data in sorted_convs[:10]:
            date = data.get("date", "")
            emotion = data.get("emotion", "")
            link = registry.make_link("conversation", conv_id)
            emoji = {"excited": "", "frustrated": "", "curious": "", "stuck": ""}.get(emotion, "")
            content.append(f"- {date}: {link} {emoji}")

        if len(sorted_convs) > 10:
            content.append(f"  - *...and {len(sorted_convs) - 10} more*")
        content.append("")

    return "\n".join(content)


def generate_index_note(registry: LinkRegistry, stats: dict) -> str:
    """Generate the main vault index."""
    content = [
        "---",
        "type: index",
        "tags:",
        "  - type/index",
        "---",
        "",
        "# Resurface Knowledge Vault",
        "",
        f"> Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ""
    ]

    content.append("## Statistics")
    content.append(f"- **Conversations:** {stats.get('conversations', 0)}")
    content.append(f"- **Ideas:** {stats.get('ideas', 0)}")
    content.append(f"- **Problems:** {stats.get('problems', 0)}")
    content.append(f"- **Workflows:** {stats.get('workflows', 0)}")
    content.append(f"- **Tools:** {stats.get('tools', 0)}")
    content.append(f"- **Themes:** {stats.get('themes', 0)}")
    content.append("")

    content.append("## Maps of Content")
    content.append("- [[Maps/Tools MOC|Tools Map]]")
    content.append("- [[Maps/Themes MOC|Themes Map]]")
    content.append("- [[Maps/Timeline MOC|Timeline]]")
    content.append("")

    content.append("## Browse by Category")
    content.append("### Ideas")
    content.append("- [[Ideas/Quick Wins/|Quick Wins]] - Low effort, high value")
    content.append("- [[Ideas/Validate/|Validate]] - Worth exploring further")
    content.append("- [[Ideas/Revive/|Revive]] - Old ideas worth revisiting")
    content.append("- [[Ideas/Someday/|Someday]] - Future possibilities")
    content.append("")

    content.append("### Other")
    content.append("- [[Problems/|Problems]] - Pain points and challenges")
    content.append("- [[Workflows/|Workflows]] - Processes and automations")
    content.append("- [[Generated/|Generated Ideas]] - AI-suggested projects")
    content.append("")

    content.append("## Profile")
    content.append("- [[Profile/Passion Profile|Your Passion Profile]]")
    content.append("")

    content.append("## Graph View Tips")
    content.append("1. Press `Ctrl/Cmd + G` to open Graph View")
    content.append("2. Use filters to focus:")
    content.append("   - `-#type/conversation` to hide conversations")
    content.append("   - `#category/quick_win` to show only quick wins")
    content.append("   - `#emotion/excited` to see exciting topics")
    content.append("3. Adjust depth slider to see more/fewer connections")
    content.append("")

    return "\n".join(content)


# =============================================================================
# MANIFEST & INCREMENTAL EXPORT
# =============================================================================

def get_export_manifest() -> dict:
    """Load or create export manifest tracking what's been exported."""
    manifest_path = VAULT_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {
        "exported_conversations": [],
        "last_export": None,
        "vault_version": "1.0"
    }


def save_export_manifest(manifest: dict):
    """Save export manifest."""
    manifest_path = VAULT_DIR / "_manifest.json"
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def run_export(
    include_conversations: bool = True,
    include_consolidated: bool = True,
    include_synthesized: bool = True,
    clean_export: bool = False
) -> dict:
    """
    Run full Obsidian vault export.

    Args:
        include_conversations: Export full conversation transcripts
        include_consolidated: Export consolidated ideas/problems/workflows
        include_synthesized: Export passion profile and generated ideas
        clean_export: Delete existing vault before export

    Returns:
        Dict with export statistics
    """
    import shutil

    try:
        update_status("Initializing export...", progress_pct=0)

        # Clean if requested
        if clean_export and VAULT_DIR.exists():
            shutil.rmtree(VAULT_DIR)

        # Create directory structure
        VAULT_DIR.mkdir(parents=True, exist_ok=True)

        subdirs = [
            "Maps",
            "Conversations",
            "Ideas", "Ideas/Quick Wins", "Ideas/Validate", "Ideas/Revive", "Ideas/Someday",
            "Problems",
            "Workflows",
            "Tools",
            "Themes",
            "Generated", "Generated/Intersection Ideas", "Generated/Problem Solutions",
            "Generated/Profile Based", "Generated/Time Capsule",
            "Saved",
            "Profile"
        ]
        for subdir in subdirs:
            (VAULT_DIR / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize link registry
        registry = LinkRegistry()
        stats = {
            "conversations": 0,
            "ideas": 0,
            "problems": 0,
            "workflows": 0,
            "tools": 0,
            "themes": 0,
            "generated": 0
        }

        # Load all data
        update_status("Loading data...", progress_pct=5)

        parsed = load_all_parsed() if include_conversations else []
        extractions = load_all_extractions()
        consolidated = load_consolidated() if include_consolidated else {}
        categorized = load_categorized() if include_consolidated else {}
        profile = load_passion_profile() if include_synthesized else {}
        generated = load_generated_ideas() if include_synthesized else []
        saved = load_saved_ideas() if include_synthesized else []

        # Phase 1: Register all items for linking
        update_status("Building link registry...", progress_pct=10)

        # Always register tools from ALL extractions (even if not exporting conversations)
        # This ensures tools are available for linking regardless of export options
        for conv_id, ext_data in extractions.items():
            ext = ext_data.get("extraction", {})
            tools = ext.get("tools_explored", [])
            for tool in tools:
                registry.tools[tool]["conversations"].append(conv_id)
            # Track co-occurrence
            for i, t1 in enumerate(tools):
                for t2 in tools[i+1:]:
                    registry.tool_cooccurrence[t1][t2] += 1
                    registry.tool_cooccurrence[t2][t1] += 1

        # Also register tools from passion profile's tool_expertise
        for tool_exp in profile.get("tool_expertise", []):
            tool_name = tool_exp.get("tool", "")
            if tool_name:
                # Access to initialize (defaultdict creates entry)
                _ = registry.tools[tool_name]

        # Register conversations (with extraction data merged) - only if including conversations
        for conv in parsed:
            conv_id = conv.get("id", "")
            ext = extractions.get(conv_id, {}).get("extraction", {})
            merged = {
                **conv,
                "tools_explored": ext.get("tools_explored", []),
                "project_ideas": ext.get("project_ideas", []),
                "problems": ext.get("problems", []),
                "workflows": ext.get("workflows", []),
                "emotion": ext.get("emotional_signals", {}).get("tone", "neutral")
            }
            registry.register_conversation(conv_id, merged)

        # Register ideas (prefer categorized, fall back to consolidated)
        ideas_list = categorized.get("ideas", consolidated.get("idea_clusters", []))
        for idea in ideas_list:
            category = idea.get("category", "someday")
            registry.register_idea(idea, category)

        # Register problems
        for problem in consolidated.get("problem_clusters", []):
            registry.register_problem(problem)

        # Register workflows
        for workflow in consolidated.get("workflow_clusters", []):
            registry.register_workflow(workflow)

        # Register themes
        for theme in profile.get("core_themes", []):
            registry.register_theme(theme)

        # Calculate total items for progress
        total_items = (
            len(parsed) +
            len(ideas_list) +
            len(consolidated.get("problem_clusters", [])) +
            len(consolidated.get("workflow_clusters", [])) +
            len(registry.tools) +
            len(profile.get("core_themes", [])) +
            len(generated)
        )
        current = 0

        # Phase 2: Generate and write notes

        # Write conversation notes
        update_status("Exporting conversations...", progress_pct=15)
        for conv in parsed:
            conv_id = conv.get("id", "")
            created = conv.get("created", "0000-00-00")
            month = created[:7] if created else "0000-00"
            slug = f"{slugify(conv.get('title', 'untitled'))}-{conv_id[:8]}"

            # Create month directory
            month_dir = VAULT_DIR / "Conversations" / month
            month_dir.mkdir(parents=True, exist_ok=True)

            # Generate and write note
            extraction = extractions.get(conv_id, {})
            note_content = generate_conversation_note(conv, extraction, registry)

            note_path = month_dir / f"{slug}.md"
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["conversations"] += 1
            current += 1
            if current % 20 == 0:
                pct = 15 + (current / max(total_items, 1) * 50)
                update_status(f"Exported {current} items...", progress_pct=pct)

        # Write idea notes
        update_status("Exporting ideas...", progress_pct=45)
        # Map category keys to folder names
        category_folders = {
            "quick_win": "Quick Wins",
            "validate": "Validate",
            "revive": "Revive",
            "someday": "Someday"
        }
        for idea in ideas_list:
            name = idea.get("name", "")
            if not name:
                continue

            category = idea.get("category", "someday")
            category_folder = category_folders.get(category, "Someday")

            note_content = generate_idea_note(idea, registry)
            note_path = VAULT_DIR / "Ideas" / category_folder / f"{slugify(name)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["ideas"] += 1
            current += 1

        # Write problem notes
        update_status("Exporting problems...", progress_pct=55)
        for problem in consolidated.get("problem_clusters", []):
            name = problem.get("name", "")
            if not name:
                continue

            note_content = generate_problem_note(problem, registry)
            note_path = VAULT_DIR / "Problems" / f"{slugify(name)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["problems"] += 1
            current += 1

        # Write workflow notes
        update_status("Exporting workflows...", progress_pct=60)
        for workflow in consolidated.get("workflow_clusters", []):
            name = workflow.get("name", "")
            if not name:
                continue

            note_content = generate_workflow_note(workflow, registry)
            note_path = VAULT_DIR / "Workflows" / f"{slugify(name)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["workflows"] += 1
            current += 1

        # Write tool notes
        update_status("Exporting tool pages...", progress_pct=65)
        for tool in registry.tools:
            note_content = generate_tool_note(tool, registry)
            note_path = VAULT_DIR / "Tools" / f"{slugify(tool)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["tools"] += 1
            current += 1

        # Write theme notes
        update_status("Exporting themes...", progress_pct=75)
        for theme in profile.get("core_themes", []):
            note_content = generate_theme_note(theme, registry)
            name = theme.get("theme", "unknown")
            note_path = VAULT_DIR / "Themes" / f"{slugify(name)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["themes"] += 1
            current += 1

        # Write generated ideas
        update_status("Exporting generated ideas...", progress_pct=80)
        for idea in generated:
            strategy = idea.get("strategy", "other")
            strategy_folders = {
                "intersection": "Intersection Ideas",
                "problem_solution": "Problem Solutions",
                "profile_based": "Profile Based",
                "time_capsule": "Time Capsule"
            }
            folder = strategy_folders.get(strategy, "Other")

            note_content = generate_generated_idea_note(idea, registry)
            name = idea.get("name", "untitled")
            note_path = VAULT_DIR / "Generated" / folder / f"{slugify(name)}.md"

            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(note_content)

            stats["generated"] += 1
            current += 1

        # Write passion profile
        if profile:
            update_status("Exporting passion profile...", progress_pct=85)
            profile_content = generate_passion_profile_note(profile, registry)
            with open(VAULT_DIR / "Profile" / "Passion Profile.md", 'w', encoding='utf-8') as f:
                f.write(profile_content)

        # Write MOCs
        update_status("Generating Maps of Content...", progress_pct=90)

        with open(VAULT_DIR / "Maps" / "Tools MOC.md", 'w', encoding='utf-8') as f:
            f.write(generate_tools_moc(registry))

        with open(VAULT_DIR / "Maps" / "Themes MOC.md", 'w', encoding='utf-8') as f:
            f.write(generate_themes_moc(registry, profile))

        with open(VAULT_DIR / "Maps" / "Timeline MOC.md", 'w', encoding='utf-8') as f:
            f.write(generate_timeline_moc(registry))

        # Write index
        update_status("Generating index...", progress_pct=95)
        with open(VAULT_DIR / "_Index.md", 'w', encoding='utf-8') as f:
            f.write(generate_index_note(registry, stats))

        # Create .obsidian config with default graph settings
        # This hides conversations from graph by default
        obsidian_config_dir = VAULT_DIR / ".obsidian"
        obsidian_config_dir.mkdir(exist_ok=True)

        graph_config = {
            "collapse-filter": False,
            "search": "-path:Conversations",
            "showTags": False,
            "showAttachments": False,
            "hideUnresolved": False,
            "showOrphans": True
        }

        with open(obsidian_config_dir / "graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_config, f, indent=2)

        # Save manifest
        manifest = {
            "exported_conversations": [c.get("id") for c in parsed],
            "last_export": datetime.now().isoformat(),
            "vault_version": "1.0",
            "stats": stats
        }
        save_export_manifest(manifest)

        update_status(
            f"Export complete! {stats['conversations']} conversations, "
            f"{stats['ideas']} ideas, {stats['tools']} tools",
            progress_pct=100,
            complete=True
        )

        return stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_status(f"Export failed: {str(e)}", error=True)
        return {}


def run_incremental_export() -> dict:
    """
    Export only new/changed items since last export.

    Checks manifest for previously exported conversation IDs
    and only processes new ones. For simplicity, runs full export
    if there are new items (tool/theme pages may need updating).
    """
    try:
        manifest = get_export_manifest()
        exported_ids = set(manifest.get("exported_conversations", []))

        # Load current parsed conversations
        parsed = load_all_parsed()
        current_ids = {c.get("id") for c in parsed}

        # Find new conversations
        new_ids = current_ids - exported_ids

        if not new_ids:
            update_status("No new conversations to export", progress_pct=100, complete=True)
            return {"new_conversations": 0}

        update_status(f"Found {len(new_ids)} new conversations, running export...", progress_pct=5)

        # Run full export (tool pages need to be regenerated anyway)
        stats = run_export(clean_export=False)
        stats["new_conversations"] = len(new_ids)

        return stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_status(f"Incremental export failed: {str(e)}", error=True)
        return {}


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Resurface data to Obsidian vault")
    parser.add_argument("--clean", action="store_true", help="Delete existing vault before export")
    parser.add_argument("--incremental", action="store_true", help="Only export new items")
    parser.add_argument("--no-conversations", action="store_true", help="Skip conversation export")
    parser.add_argument("--no-consolidated", action="store_true", help="Skip consolidated data")
    parser.add_argument("--no-synthesized", action="store_true", help="Skip synthesized data")
    args = parser.parse_args()

    if args.incremental:
        stats = run_incremental_export()
    else:
        stats = run_export(
            include_conversations=not args.no_conversations,
            include_consolidated=not args.no_consolidated,
            include_synthesized=not args.no_synthesized,
            clean_export=args.clean
        )

    print(f"\nExport complete: {stats}")
