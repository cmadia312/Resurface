#!/usr/bin/env python3
"""
Extraction runner - orchestrates conversation extraction.

Handles batching, progress tracking, resumability, and rate limiting.

Usage:
    python runner.py --status           # Show extraction status
    python runner.py --count 10         # Extract next 10 conversations
    python runner.py --all              # Extract all remaining
    python runner.py --id <uuid>        # Extract specific conversation
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from config import load_config, validate_config
from extractor import extract_conversation


PARSED_DIR = Path("data/parsed")
EXTRACTIONS_DIR = Path("data/extractions")
MANIFEST_FILE = PARSED_DIR / "_manifest.json"
STATUS_FILE = Path("data/extraction_status.json")


def update_status(message: str, progress_pct: float,
                  current: int, total: int,
                  elapsed_sec: float, eta_sec: float | None,
                  complete: bool = False, error: bool = False):
    """Atomically update extraction status file."""
    status = {
        "message": message,
        "progress": progress_pct,
        "current": current,
        "total": total,
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


def ensure_dirs():
    """Ensure output directory exists."""
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> dict:
    """Load the parsed conversations manifest."""
    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest not found at {MANIFEST_FILE}")
        print("Run parser.py first to generate parsed conversations.")
        sys.exit(1)

    with open(MANIFEST_FILE, 'r') as f:
        return json.load(f)


def get_extraction_status() -> dict:
    """
    Get current extraction status.

    Returns dict with total, extracted, remaining, errors counts.
    """
    manifest = load_manifest()
    total = len(manifest.get('conversations', []))

    # Count existing extraction files
    extracted = 0
    errors = 0

    for conv in manifest.get('conversations', []):
        conv_id = conv['id']
        extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"

        if extraction_file.exists():
            extracted += 1
            # Check if it's an error
            with open(extraction_file, 'r') as f:
                extraction = json.load(f)
                if extraction.get('extraction', {}).get('error'):
                    errors += 1

    return {
        "total_conversations": total,
        "extracted": extracted,
        "remaining": total - extracted,
        "errors": errors
    }


def get_pending_conversations(limit: int = None) -> list:
    """
    Get conversations that haven't been extracted yet.
    Returns newest conversations first.

    Args:
        limit: Maximum number to return (None for all)

    Returns:
        List of conversation metadata dicts, sorted by date (newest first)
    """
    manifest = load_manifest()
    pending = []

    for conv in manifest.get('conversations', []):
        conv_id = conv['id']
        extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"

        if not extraction_file.exists():
            pending.append(conv)

    # Sort by date, newest first
    pending.sort(key=lambda x: x.get('created', ''), reverse=True)

    # Apply limit after sorting
    if limit:
        pending = pending[:limit]

    return pending


def load_conversation(conv_id: str) -> dict | None:
    """Load a parsed conversation by ID."""
    conv_file = PARSED_DIR / f"{conv_id}.json"
    if not conv_file.exists():
        return None

    with open(conv_file, 'r') as f:
        return json.load(f)


def save_extraction(conv_id: str, conversation: dict, extraction: dict, model: str) -> None:
    """Save extraction result to file."""
    output = {
        "conversation_id": conv_id,
        "conversation_title": conversation.get('title'),
        "conversation_date": conversation.get('created', '')[:10] if conversation.get('created') else None,
        "extraction": extraction,
        "model_used": model,
        "extracted_at": datetime.now(tz=timezone.utc).isoformat()
    }

    output_file = EXTRACTIONS_DIR / f"{conv_id}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def run_extraction(count: int = None, conversation_ids: list = None) -> dict:
    """
    Run extraction on conversations.

    Args:
        count: Number of pending conversations to process (None for all)
        conversation_ids: Specific conversation IDs to process

    Returns:
        Dict with processed, empty, errors counts
    """
    ensure_dirs()
    config = load_config()

    # Validate config
    is_valid, error = validate_config(config)
    if not is_valid:
        print(f"Configuration error: {error}")
        update_status(f"Configuration error: {error}", 0, 0, 0, 0, None, error=True)
        return {"processed": 0, "empty": 0, "errors": 1, "error_ids": ["config"]}

    # Determine which conversations to process
    if conversation_ids:
        # Specific IDs provided
        to_process = [{"id": cid} for cid in conversation_ids]
    else:
        # Get pending conversations
        to_process = get_pending_conversations(limit=count)

    if not to_process:
        print("No conversations to process.")
        update_status("No conversations to process.", 100, 0, 0, 0, 0, complete=True)
        return {"processed": 0, "empty": 0, "errors": 0, "error_ids": []}

    # Rate limiting setup
    rpm = config.get('requests_per_minute', 60)
    delay = 60.0 / rpm

    # Stats
    stats = {
        "processed": 0,
        "empty": 0,
        "errors": 0,
        "error_ids": []
    }

    total = len(to_process)
    model = config.get('model')

    print(f"Extracting {total} conversations using {model}...")
    print(f"Rate limit: {rpm} requests/minute ({delay:.1f}s between requests)")
    print()

    # Timing
    start_time = time.time()
    update_status(f"Starting extraction of {total} conversations...", 0, 0, total, 0, None)

    for i, conv_meta in enumerate(to_process):
        conv_id = conv_meta['id']

        # Skip if already extracted (for specific ID mode)
        extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"
        if extraction_file.exists() and not conversation_ids:
            continue

        # Load conversation
        conversation = load_conversation(conv_id)
        if not conversation:
            print(f"[{i+1}/{total}] ERROR: Could not load {conv_id}")
            stats["errors"] += 1
            stats["error_ids"].append(conv_id)
            continue

        title = conversation.get('title', 'Untitled')[:50]
        print(f"[{i+1}/{total}] Extracting \"{title}\"...")

        # Calculate timing and update status before extraction
        elapsed = time.time() - start_time
        items_done = i
        if items_done > 0:
            avg_per_item = elapsed / items_done
            eta = avg_per_item * (total - items_done)
        else:
            eta = None

        progress_pct = (i / total) * 100
        update_status(f"Extracting: {title}", progress_pct, i + 1, total, elapsed, eta)

        # Extract
        extraction = extract_conversation(conversation, config)

        # Save result
        save_extraction(conv_id, conversation, extraction, model)
        stats["processed"] += 1

        # Track empty/error
        if extraction.get('empty'):
            stats["empty"] += 1
        elif extraction.get('error'):
            stats["errors"] += 1
            stats["error_ids"].append(conv_id)

        # Rate limiting (skip on last item)
        if i < total - 1:
            time.sleep(delay)

    # Final timing
    final_elapsed = time.time() - start_time

    # Print summary
    print()
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Processed:     {stats['processed']}")
    print(f"Empty:         {stats['empty']}")
    print(f"Errors:        {stats['errors']}")
    print(f"Total time:    {final_elapsed:.1f}s")

    if stats["error_ids"]:
        print(f"\nError IDs: {stats['error_ids'][:5]}")
        if len(stats["error_ids"]) > 5:
            print(f"  ... and {len(stats['error_ids']) - 5} more")

    # Final status update
    update_status(
        f"Complete: {stats['processed']} processed, {stats['empty']} empty, {stats['errors']} errors",
        100, total, total, final_elapsed, 0, complete=True
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run conversation extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --status           Show extraction progress
  python runner.py --count 10         Extract next 10 conversations
  python runner.py --all              Extract all remaining conversations
  python runner.py --id abc123        Extract specific conversation
        """
    )

    parser.add_argument('--status', action='store_true',
                        help='Show extraction status')
    parser.add_argument('--count', type=int, metavar='N',
                        help='Extract N pending conversations')
    parser.add_argument('--all', action='store_true',
                        help='Extract all remaining conversations')
    parser.add_argument('--id', type=str, metavar='UUID',
                        help='Extract specific conversation by ID')

    args = parser.parse_args()

    # Default to status if no args
    if not any([args.status, args.count, args.all, args.id]):
        args.status = True

    if args.status:
        status = get_extraction_status()
        print("Extraction Status")
        print("-" * 30)
        print(f"Total conversations: {status['total_conversations']}")
        print(f"Extracted:           {status['extracted']}")
        print(f"Remaining:           {status['remaining']}")
        print(f"Errors:              {status['errors']}")

        if status['remaining'] > 0:
            pct = (status['extracted'] / status['total_conversations']) * 100
            print(f"\nProgress: {pct:.1f}%")

    elif args.id:
        run_extraction(conversation_ids=[args.id])

    elif args.count:
        run_extraction(count=args.count)

    elif args.all:
        run_extraction()


if __name__ == "__main__":
    main()
