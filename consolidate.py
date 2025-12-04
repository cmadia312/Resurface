#!/usr/bin/env python3
"""
LLM-based consolidation for Resurface.

Groups similar/duplicate items across extractions using LLM semantic understanding.
Processes in batches and recursively merges until stable.
"""
import json
import time
import os
import tempfile
from pathlib import Path
from datetime import datetime

from config import load_config
from llm_provider import call_llm, make_array_schema
from schemas import ConsolidatedIdea, ConsolidatedProblem, ConsolidatedWorkflow

EXTRACTIONS_DIR = Path("data/extractions")
CONSOLIDATED_DIR = Path("data/consolidated")
STATUS_FILE = Path("data/consolidation_status.json")


def update_status(message: str, progress_pct: float | None = None,
                  elapsed_sec: float | None = None, eta_sec: float | None = None,
                  complete: bool = False, error: bool = False):
    """Atomically update consolidation status file."""
    status = {
        "message": message,
        "progress": progress_pct,
        "elapsed_seconds": elapsed_sec,
        "eta_seconds": eta_sec,
        "complete": complete,
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid()
    }
    STATUS_FILE.parent.mkdir(exist_ok=True)

    # Atomic write: write to temp, then rename
    fd, tmp_path = tempfile.mkstemp(dir=STATUS_FILE.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
        os.replace(tmp_path, STATUS_FILE)  # Atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

# Prompts for each item type
CONSOLIDATION_PROMPTS = {
    "ideas": """Here are project ideas extracted from multiple conversations over time.
Group them into unique concepts—merge duplicates and near-duplicates that represent the same underlying idea.

For each unique concept, return:
- name: A clear consolidated name
- description: 2-3 sentence synthesis of all mentions
- occurrences: How many times it appeared
- date_range: [earliest_mention, latest_mention]
- evolution: Did it get more specific over time? Note any progression.
- source_ids: List of conversation_ids that contained this idea
- motivations: Combined list of motivations from all mentions
- detail_levels: List of detail levels from each mention

Return as a JSON array. Only return valid JSON, no other text.""",

    "problems": """Here are problems/frustrations extracted from multiple conversations over time.
Group them into unique themes—merge duplicates and near-duplicates that represent the same underlying issue.

For each unique theme, return:
- name: A clear consolidated name
- description: 2-3 sentence synthesis of all mentions
- occurrences: How many times it appeared
- date_range: [earliest_mention, latest_mention]
- source_ids: List of conversation_ids that contained this problem
- contexts: Combined list of contexts from all mentions

Return as a JSON array. Only return valid JSON, no other text.""",

    "workflows": """Here are workflows/automations extracted from multiple conversations over time.
Group them into unique concepts—merge duplicates and near-duplicates that represent the same workflow.

For each unique concept, return:
- name: A clear consolidated name
- description: 2-3 sentence synthesis of all mentions
- occurrences: How many times it appeared
- date_range: [earliest_mention, latest_mention]
- source_ids: List of conversation_ids that contained this workflow
- statuses: List of statuses from each mention (exploring/building/optimizing)

Return as a JSON array. Only return valid JSON, no other text.""",
}


def load_all_extractions() -> list[dict]:
    """Load all successful extractions from disk."""
    extractions = []

    for file_path in EXTRACTIONS_DIR.glob("*.json"):
        if file_path.name == "_manifest.json":
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Skip failed extractions
        extraction = data.get("extraction", {})
        if extraction.get("error") or extraction.get("empty"):
            continue

        extractions.append(data)

    return extractions


def collect_items(extractions: list[dict], item_type: str) -> list[dict]:
    """
    Collect all items of a given type from extractions.

    item_type: 'ideas', 'problems', or 'workflows'
    """
    items = []

    field_map = {
        "ideas": "project_ideas",
        "problems": "problems",
        "workflows": "workflows"
    }

    field_name = field_map.get(item_type)
    if not field_name:
        raise ValueError(f"Unknown item type: {item_type}")

    for ext in extractions:
        conv_id = ext.get("conversation_id")
        conv_date = ext.get("conversation_date")
        conv_title = ext.get("conversation_title")
        extraction = ext.get("extraction", {})

        for item in extraction.get(field_name, []):
            # Add source metadata to each item
            enriched = {
                **item,
                "_source_id": conv_id,
                "_source_date": conv_date,
                "_source_title": conv_title
            }
            items.append(enriched)

    return items


def collect_tools(extractions: list[dict]) -> dict[str, int]:
    """Collect tool frequency across all extractions."""
    tool_counts = {}

    for ext in extractions:
        extraction = ext.get("extraction", {})
        for tool in extraction.get("tools_explored", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return tool_counts


def collect_emotions(extractions: list[dict]) -> list[dict]:
    """Collect emotional signals with dates for timeline."""
    emotions = []

    for ext in extractions:
        conv_id = ext.get("conversation_id")
        conv_date = ext.get("conversation_date")
        conv_title = ext.get("conversation_title")
        extraction = ext.get("extraction", {})

        signals = extraction.get("emotional_signals", {})
        if signals and signals.get("tone"):
            emotions.append({
                "conversation_id": conv_id,
                "date": conv_date,
                "title": conv_title,
                "tone": signals.get("tone"),
                "notes": signals.get("notes", "")
            })

    # Sort by date
    emotions.sort(key=lambda x: x.get("date", ""))
    return emotions


def call_llm_for_consolidation(items: list[dict], item_type: str, config: dict) -> list[dict]:
    """Call LLM to consolidate a batch of items."""
    provider = config.get('api_provider', 'openai')

    prompt = CONSOLIDATION_PROMPTS.get(item_type)
    if not prompt:
        raise ValueError(f"No consolidation prompt for item type: {item_type}")

    # Format items for the prompt
    items_text = json.dumps(items, indent=2)
    full_prompt = f"{prompt}\n\nItems to consolidate:\n{items_text}"

    # Map item type to schema for Ollama structured output
    schema_map = {
        "ideas": ConsolidatedIdea,
        "problems": ConsolidatedProblem,
        "workflows": ConsolidatedWorkflow,
    }

    # Determine schema and message format based on provider
    if provider == 'openai':
        messages = [{"role": "user", "content": full_prompt}]
        system_prompt = "You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions."
        schema = None
    elif provider == 'ollama':
        messages = [{
            "role": "user",
            "content": f"You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions.\n\n{full_prompt}"
        }]
        system_prompt = None
        schema = make_array_schema(schema_map.get(item_type, ConsolidatedIdea))
    else:
        # Anthropic
        messages = [{"role": "user", "content": full_prompt}]
        system_prompt = None
        schema = None

    response_text = call_llm(
        messages,
        config,
        schema=schema,
        max_tokens=8192,
        system_prompt=system_prompt
    )

    # Parse response
    return parse_json_array(response_text)


def parse_json_array(text: str) -> list[dict]:
    """Parse JSON array from LLM response."""
    import re

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            result = json.loads(code_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding array in text
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"Warning: Could not parse JSON array from response")
    return []


def consolidate_batch(items: list[dict], item_type: str, config: dict) -> list[dict]:
    """
    Send batch of items to LLM for consolidation.

    Args:
        items: List of items to consolidate
        item_type: 'ideas', 'problems', or 'workflows'
        config: API configuration

    Returns:
        List of consolidated items
    """
    if not items:
        return []

    print(f"  Consolidating batch of {len(items)} {item_type}...")

    result = call_llm_for_consolidation(items, item_type, config)

    # Respect rate limit
    rate_limit = config.get('requests_per_minute', 20)
    delay = 60.0 / rate_limit
    time.sleep(delay)

    return result


def consolidate_batch_with_retry(items: list[dict], item_type: str, config: dict,
                                  max_retries: int = 2) -> list[dict]:
    """
    Try consolidating with progressively smaller batches on failure.

    If parsing fails and batch is large enough, split in half and retry recursively.
    On complete failure, returns original items to prevent data loss.
    """
    if not items:
        return []

    print(f"  Consolidating batch of {len(items)} {item_type}...")

    # Try the LLM call
    result = call_llm_for_consolidation(items, item_type, config)

    # Respect rate limit
    rate_limit = config.get('requests_per_minute', 20)
    delay = 60.0 / rate_limit
    time.sleep(delay)

    if result:  # Successfully parsed
        return result

    # Failed - try splitting if batch is large enough
    if len(items) <= 5:  # Too small to split further
        print(f"  Warning: Cannot consolidate batch of {len(items)} items, keeping originals")
        return items  # Return original items unmerged

    print(f"  Retry: Splitting batch of {len(items)} into smaller chunks")
    mid = len(items) // 2
    left = consolidate_batch_with_retry(items[:mid], item_type, config, max_retries - 1)
    right = consolidate_batch_with_retry(items[mid:], item_type, config, max_retries - 1)

    return left + right


def consolidate_all(items: list[dict], item_type: str, config: dict, batch_size: int = 25,
                    progress_callback=None, progress_range: tuple = (0, 100)) -> list[dict]:
    """
    Recursively consolidate items until stable.

    1. Process in batches through consolidate_batch()
    2. If output > batch_size items, run consolidation again
    3. Repeat until stable

    Args:
        progress_callback: Optional function(message, progress_pct) to report progress
        progress_range: Tuple of (start_pct, end_pct) for this consolidation's progress range
    """
    if not items:
        return []

    current_items = items
    iteration = 1
    start_pct, end_pct = progress_range

    while True:
        print(f"\nConsolidation iteration {iteration} for {item_type}: {len(current_items)} items")

        # If we have few enough items, consolidate in one batch
        if len(current_items) <= batch_size:
            if progress_callback:
                progress_callback(f"Final pass: {len(current_items)} {item_type}", end_pct - 2)
            consolidated = consolidate_batch_with_retry(current_items, item_type, config)
            print(f"  -> Consolidated to {len(consolidated)} unique {item_type}")
            return consolidated

        # Process in batches
        batches = [current_items[i:i+batch_size] for i in range(0, len(current_items), batch_size)]
        consolidated = []

        for i, batch in enumerate(batches):
            # Calculate progress within this type's range
            # Use iteration to reduce progress range for subsequent iterations
            iter_weight = 1.0 / iteration  # First iteration gets full range, subsequent less
            batch_progress = (i + 1) / len(batches)
            current_pct = start_pct + (batch_progress * (end_pct - start_pct) * iter_weight)

            if progress_callback:
                progress_callback(
                    f"{item_type.capitalize()} iter {iteration}: batch {i+1}/{len(batches)}",
                    current_pct
                )

            print(f"  Processing batch {i+1}/{len(batches)}...")
            batch_result = consolidate_batch_with_retry(batch, item_type, config)
            consolidated.extend(batch_result)

        print(f"  -> Reduced from {len(current_items)} to {len(consolidated)} items")

        # Check if we've stabilized
        if len(consolidated) >= len(current_items):
            print(f"  -> Stabilized (no further reduction possible)")
            return consolidated

        # Continue consolidating the merged results
        current_items = consolidated
        iteration += 1

        # Safety limit
        if iteration > 10:
            print(f"  -> Reached iteration limit, stopping")
            return consolidated


def run_consolidation(batch_size: int = 25) -> dict:
    """
    Run full consolidation pipeline.

    Returns consolidated data dict.
    """
    def calc_eta(elapsed: float, progress: float) -> float | None:
        """Calculate ETA based on elapsed time and progress percentage."""
        if progress <= 0:
            return None
        total_estimated = elapsed / (progress / 100)
        return total_estimated - elapsed

    try:
        start_time = time.time()
        update_status("Starting consolidation...", progress_pct=0, elapsed_sec=0, eta_sec=None)
        config = load_config()

        print("Loading extractions...")
        extractions = load_all_extractions()
        print(f"Loaded {len(extractions)} successful extractions")

        if not extractions:
            print("No extractions to consolidate")
            update_status("No extractions to consolidate", elapsed_sec=time.time() - start_time, complete=True)
            return {}

        # Collect all items
        print("\nCollecting items from extractions...")
        elapsed = time.time() - start_time
        update_status(f"Collecting items from {len(extractions)} extractions...", progress_pct=5, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 5))
        all_ideas = collect_items(extractions, "ideas")
        all_problems = collect_items(extractions, "problems")
        all_workflows = collect_items(extractions, "workflows")
        tool_frequency = collect_tools(extractions)
        emotional_timeline = collect_emotions(extractions)

        print(f"  Ideas: {len(all_ideas)}")
        print(f"  Problems: {len(all_problems)}")
        print(f"  Workflows: {len(all_workflows)}")
        print(f"  Unique tools: {len(tool_frequency)}")
        print(f"  Emotional signals: {len(emotional_timeline)}")

        total_items = len(all_ideas) + len(all_problems) + len(all_workflows)
        elapsed = time.time() - start_time
        update_status(f"Found {total_items} items to consolidate", progress_pct=10, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 10))

        # Consolidate each type
        consolidated = {
            "idea_clusters": [],
            "problem_clusters": [],
            "workflow_clusters": [],
            "tool_frequency": tool_frequency,
            "emotional_timeline": emotional_timeline,
            "metadata": {
                "source_extractions": len(extractions),
                "consolidated_at": datetime.now().isoformat(),
                "raw_counts": {
                    "ideas": len(all_ideas),
                    "problems": len(all_problems),
                    "workflows": len(all_workflows)
                }
            }
        }

        # Progress callback that updates status with current elapsed time
        def make_progress_callback():
            def callback(message, progress_pct):
                elapsed = time.time() - start_time
                update_status(message, progress_pct=progress_pct, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, progress_pct))
            return callback

        progress_cb = make_progress_callback()

        if all_ideas:
            print("\n--- Consolidating Ideas ---")
            elapsed = time.time() - start_time
            update_status(f"Consolidating {len(all_ideas)} ideas...", progress_pct=15, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 15))
            consolidated["idea_clusters"] = consolidate_all(
                all_ideas, "ideas", config, batch_size,
                progress_callback=progress_cb, progress_range=(15, 40)
            )
            elapsed = time.time() - start_time
            update_status(f"Ideas: {len(all_ideas)} -> {len(consolidated['idea_clusters'])} clusters", progress_pct=40, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 40))

        if all_problems:
            print("\n--- Consolidating Problems ---")
            elapsed = time.time() - start_time
            update_status(f"Consolidating {len(all_problems)} problems...", progress_pct=45, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 45))
            consolidated["problem_clusters"] = consolidate_all(
                all_problems, "problems", config, batch_size,
                progress_callback=progress_cb, progress_range=(45, 70)
            )
            elapsed = time.time() - start_time
            update_status(f"Problems: {len(all_problems)} -> {len(consolidated['problem_clusters'])} clusters", progress_pct=70, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 70))

        if all_workflows:
            print("\n--- Consolidating Workflows ---")
            elapsed = time.time() - start_time
            update_status(f"Consolidating {len(all_workflows)} workflows...", progress_pct=75, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 75))
            consolidated["workflow_clusters"] = consolidate_all(
                all_workflows, "workflows", config, batch_size,
                progress_callback=progress_cb, progress_range=(75, 95)
            )
            elapsed = time.time() - start_time
            update_status(f"Workflows: {len(all_workflows)} -> {len(consolidated['workflow_clusters'])} clusters", progress_pct=95, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 95))

        # Save consolidated data
        CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

        output_path = CONSOLIDATED_DIR / "consolidated.json"
        with open(output_path, 'w') as f:
            json.dump(consolidated, f, indent=2)

        final_elapsed = time.time() - start_time
        print(f"\n=== Consolidation Complete ===")
        print(f"Unique idea clusters: {len(consolidated['idea_clusters'])}")
        print(f"Unique problem clusters: {len(consolidated['problem_clusters'])}")
        print(f"Unique workflow clusters: {len(consolidated['workflow_clusters'])}")
        print(f"Total time: {final_elapsed:.1f}s")
        print(f"\nSaved to: {output_path}")

        update_status(
            f"Complete: {len(consolidated['idea_clusters'])} ideas, "
            f"{len(consolidated['problem_clusters'])} problems, "
            f"{len(consolidated['workflow_clusters'])} workflows",
            progress_pct=100, elapsed_sec=final_elapsed, eta_sec=0, complete=True
        )

        return consolidated

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        update_status(error_msg, elapsed_sec=time.time() - start_time if 'start_time' in dir() else None, error=True)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate extractions using LLM")
    parser.add_argument("--batch-size", type=int, default=25, help="Items per batch (default: 25)")
    args = parser.parse_args()

    run_consolidation(batch_size=args.batch_size)
