#!/usr/bin/env python3
"""
Data management utilities for Resurface.

Provides status reporting and reset functionality for processed data.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
EXTRACTIONS_DIR = DATA_DIR / "extractions"
PARSED_DIR = DATA_DIR / "parsed"
CONSOLIDATED_DIR = DATA_DIR / "consolidated"
SYNTHESIZED_DIR = DATA_DIR / "synthesized"
SYNTHESIS_STATUS_FILE = DATA_DIR / "synthesis_status.json"


def get_data_status() -> dict:
    """
    Return counts and status of all data stages.

    Returns dict with:
        - parsed_count: Number of parsed conversations
        - extraction_count: Number of completed extractions
        - consolidation_status: "Not run" or "Last run: YYYY-MM-DD HH:MM"
        - categorization_status: "Not run" or "Last run: YYYY-MM-DD HH:MM"
        - synthesis_status: "Not run" or "Last run: YYYY-MM-DD HH:MM"
    """
    status = {
        "parsed_count": 0,
        "extraction_count": 0,
        "consolidation_status": "Not run",
        "categorization_status": "Not run",
        "synthesis_status": "Not run",
    }

    # Count parsed conversations from manifest
    manifest_path = PARSED_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
            status["parsed_count"] = manifest.get("parsed", 0)

    # Count extractions (exclude files starting with _)
    if EXTRACTIONS_DIR.exists():
        status["extraction_count"] = len([
            f for f in EXTRACTIONS_DIR.glob("*.json")
            if not f.name.startswith("_")
        ])

    # Check consolidation status
    consolidated_path = CONSOLIDATED_DIR / "consolidated.json"
    if consolidated_path.exists():
        mtime = datetime.fromtimestamp(consolidated_path.stat().st_mtime)
        status["consolidation_status"] = f"Last run: {mtime.strftime('%Y-%m-%d %H:%M')}"

    # Check categorization status
    categorized_path = CONSOLIDATED_DIR / "categorized.json"
    if categorized_path.exists():
        mtime = datetime.fromtimestamp(categorized_path.stat().st_mtime)
        status["categorization_status"] = f"Last run: {mtime.strftime('%Y-%m-%d %H:%M')}"

    # Check synthesis status
    profile_path = SYNTHESIZED_DIR / "passion_profile.json"
    if profile_path.exists():
        mtime = datetime.fromtimestamp(profile_path.stat().st_mtime)
        status["synthesis_status"] = f"Last run: {mtime.strftime('%Y-%m-%d %H:%M')}"

    return status


def reset_extractions() -> dict:
    """
    Delete all extraction files in data/extractions/.

    Returns dict with:
        - deleted: Number of files deleted
        - type: "extractions"
    """
    count = 0
    if EXTRACTIONS_DIR.exists():
        for f in EXTRACTIONS_DIR.glob("*.json"):
            f.unlink()
            count += 1
    return {"deleted": count, "type": "extractions"}


def reset_all_processed() -> dict:
    """
    Delete all processed data: parsed, extractions, consolidated, categorized, AND synthesized.

    Preserves:
        - data/conversations.json (original export)
        - data/chatgpt_conversations.json (uploaded ChatGPT export)
        - data/claude_conversations.json (uploaded Claude export)
        - config.json (API settings)

    Returns dict with:
        - parsed: True if parsed directory was deleted
        - extractions: True if extractions directory was deleted
        - consolidated: True if consolidated.json was deleted
        - categorized: True if categorized.json was deleted
        - synthesized: True if synthesized directory was deleted
    """
    result = {"parsed": False, "extractions": False, "consolidated": False, "categorized": False, "synthesized": False}

    # Delete parsed directory
    if PARSED_DIR.exists():
        shutil.rmtree(PARSED_DIR)
        result["parsed"] = True

    # Delete extractions directory
    if EXTRACTIONS_DIR.exists():
        shutil.rmtree(EXTRACTIONS_DIR)
        result["extractions"] = True

    # Delete consolidated.json
    consolidated_path = CONSOLIDATED_DIR / "consolidated.json"
    if consolidated_path.exists():
        consolidated_path.unlink()
        result["consolidated"] = True

    # Delete categorized.json
    categorized_path = CONSOLIDATED_DIR / "categorized.json"
    if categorized_path.exists():
        categorized_path.unlink()
        result["categorized"] = True

    # Delete synthesized directory (passion profile, generated ideas, saved ideas, developed specs)
    if SYNTHESIZED_DIR.exists():
        shutil.rmtree(SYNTHESIZED_DIR)
        result["synthesized"] = True

    # Delete all status files
    for status_file in DATA_DIR.glob("*_status.json"):
        status_file.unlink()

    return result


def format_status_markdown(status: dict) -> str:
    """Format status dict as markdown for display."""
    return f"""**Parsed conversations:** {status['parsed_count']:,}

**Extractions completed:** {status['extraction_count']:,}

**Consolidation:** {status['consolidation_status']}

**Categorization:** {status['categorization_status']}

**Synthesis:** {status.get('synthesis_status', 'Not run')}"""


if __name__ == "__main__":
    # Show current status when run directly
    status = get_data_status()
    print("Data Status:")
    print(f"  Parsed conversations: {status['parsed_count']}")
    print(f"  Extractions completed: {status['extraction_count']}")
    print(f"  Consolidation: {status['consolidation_status']}")
    print(f"  Categorization: {status['categorization_status']}")
