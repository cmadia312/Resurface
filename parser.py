#!/usr/bin/env python3
"""
Conversation export parser - extracts and cleans conversations

Supports multiple export formats:
- ChatGPT: Tree-based mapping structure from OpenAI exports
- Claude: Flat chat_messages array from Anthropic exports

Outputs normalized JSON files per conversation for downstream processing.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Stats:
    """Track parsing statistics"""
    total: int = 0
    parsed: int = 0
    skipped_trivial: int = 0
    skipped_malformed: int = 0
    total_messages: int = 0
    errors: list = field(default_factory=list)
    dates: list = field(default_factory=list)


def timestamp_to_iso(ts: float | None) -> str | None:
    """Convert Unix timestamp to ISO format string"""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return None


def linearize_conversation(mapping: dict, current_node: str) -> list[str]:
    """Walk tree backwards from current_node to root to get linear message sequence"""
    path = []
    node_id = current_node
    seen = set()  # Prevent infinite loops from malformed data

    while node_id and node_id not in seen:
        seen.add(node_id)
        node = mapping.get(node_id)
        if not node:
            break
        path.append(node_id)
        node_id = node.get('parent')

    return list(reversed(path))


def extract_text_from_parts(parts: list) -> tuple[str, bool]:
    """
    Extract text content from message parts.
    Returns (text_content, has_image)
    """
    text_pieces = []
    has_image = False

    for part in parts:
        if isinstance(part, str):
            text_pieces.append(part)
        elif isinstance(part, dict):
            # Image or other media
            if part.get('content_type') == 'image_asset_pointer' or 'asset_pointer' in part:
                has_image = True
            # Some dicts might have text content
            if 'text' in part:
                text_pieces.append(part['text'])

    return '\n'.join(text_pieces).strip(), has_image


def extract_message(node: dict) -> dict | None:
    """
    Extract clean message from a node.
    Returns None if message should be skipped.
    """
    msg = node.get('message')
    if not msg:
        return None

    author = msg.get('author', {})
    role = author.get('role')
    content = msg.get('content', {})
    content_type = content.get('content_type')
    parts = content.get('parts', [])

    # Skip system messages
    if role == 'system':
        return None

    # Skip user_editable_context (custom instructions)
    if content_type == 'user_editable_context':
        return None

    # Skip reasoning model internals
    if content_type in ('reasoning_recap', 'thoughts'):
        return None

    # Handle tool messages - summarize presence only
    if role == 'tool':
        tool_name = author.get('name', 'unknown')
        return {
            'role': 'tool',
            'tool_name': tool_name,
            'content': f'[Tool: {tool_name}]'
        }

    # Handle tether_browsing_display
    if content_type == 'tether_browsing_display':
        return {
            'role': 'assistant',
            'content': '[Web search performed]',
            'tool_name': 'web_search'
        }

    # Handle code execution results
    if content_type == 'code':
        return {
            'role': 'assistant',
            'content': '[Code executed]',
            'tool_name': 'code'
        }

    # Handle user and assistant text messages
    if role in ('user', 'assistant'):
        text, has_image = extract_text_from_parts(parts)

        # Skip empty messages
        if not text and not has_image:
            return None

        result = {
            'role': role,
            'content': text
        }
        if has_image:
            result['has_image'] = True

        return result

    return None


def process_conversation(conv: dict) -> tuple[dict | None, str | None]:
    """
    Process a single conversation.
    Returns (parsed_conversation, error_message)
    """
    conv_id = conv.get('conversation_id') or conv.get('id')

    # Validate required fields
    mapping = conv.get('mapping')
    current_node = conv.get('current_node')

    if not mapping:
        return None, f"Missing mapping in {conv_id}"
    if not current_node:
        return None, f"Missing current_node in {conv_id}"
    if current_node not in mapping:
        return None, f"current_node not in mapping for {conv_id}"

    # Linearize the conversation tree
    path = linearize_conversation(mapping, current_node)
    if len(path) < 2:
        return None, f"Path too short in {conv_id}"

    # Extract messages
    messages = []
    tools_used = set()
    has_images = False

    for node_id in path:
        node = mapping.get(node_id, {})
        extracted = extract_message(node)
        if extracted:
            messages.append(extracted)
            if extracted.get('has_image'):
                has_images = True
            if 'tool_name' in extracted:
                tools_used.add(extracted['tool_name'])

    # Count user/assistant turns (not tool messages)
    turn_count = sum(1 for m in messages if m['role'] in ('user', 'assistant'))

    # Build output
    result = {
        'id': conv_id,
        'title': conv.get('title'),
        'created': timestamp_to_iso(conv.get('create_time')),
        'updated': timestamp_to_iso(conv.get('update_time')),
        'model': conv.get('default_model_slug'),
        'message_count': len(messages),
        'turn_count': turn_count,
        'messages': messages
    }

    if has_images:
        result['has_images'] = True
    if tools_used:
        result['tools_used'] = sorted(tools_used)

    return result, None


def process_claude_conversation(conv: dict) -> tuple[dict | None, str | None]:
    """
    Process a Claude.ai conversation export.
    Returns (parsed_conversation, error_message)
    """
    conv_id = conv.get('uuid')
    chat_messages = conv.get('chat_messages', [])

    if not chat_messages:
        return None, f"No messages in {conv_id}"

    messages = []
    has_images = False

    for msg in chat_messages:
        sender = msg.get('sender', '')
        text = msg.get('text', '').strip()

        # Map Claude roles to normalized roles
        if sender == 'human':
            role = 'user'
        elif sender == 'assistant':
            role = 'assistant'
        else:
            continue  # Skip unknown sender types

        # Check for attachments (images/files)
        attachments = msg.get('attachments', [])
        files = msg.get('files', [])
        if attachments or files:
            has_images = True

        if text:
            result_msg = {'role': role, 'content': text}
            if attachments or files:
                result_msg['has_image'] = True
            messages.append(result_msg)

    turn_count = len(messages)

    result = {
        'id': conv_id,
        'title': conv.get('name') or 'Untitled',
        'created': conv.get('created_at'),
        'updated': conv.get('updated_at'),
        'model': 'claude',
        'message_count': len(messages),
        'turn_count': turn_count,
        'messages': messages
    }

    if has_images:
        result['has_images'] = True

    return result, None


def main():
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(
        description='Parse conversation exports from ChatGPT or Claude'
    )
    arg_parser.add_argument(
        '--format',
        choices=['chatgpt', 'claude'],
        default='chatgpt',
        help='Export format to parse (default: chatgpt)'
    )
    arg_parser.add_argument(
        '--input',
        default='data/conversations.json',
        help='Path to input JSON file (default: data/conversations.json)'
    )
    args = arg_parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path('data/parsed')
    export_format = args.format

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    print(f"Format: {export_format}")
    with open(input_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations")

    stats = Stats(total=len(conversations))
    manifest_entries = []

    for i, conv in enumerate(conversations):
        # Get conversation ID based on format
        if export_format == 'claude':
            conv_id = conv.get('uuid') or f'unknown_{i}'
        else:
            conv_id = conv.get('conversation_id') or conv.get('id') or f'unknown_{i}'

        try:
            # Route to appropriate parser
            if export_format == 'claude':
                result, error = process_claude_conversation(conv)
            else:
                result, error = process_conversation(conv)

            if error:
                stats.skipped_malformed += 1
                stats.errors.append({'id': conv_id, 'error': error})
                continue

            # Check trivial filter (< 4 turns)
            if result['turn_count'] < 4:
                stats.skipped_trivial += 1
                continue

            # Track dates for range
            if result['created']:
                stats.dates.append(result['created'][:10])

            # Write individual file
            output_path = output_dir / f"{conv_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Add to manifest (include source format for multi-format support)
            manifest_entries.append({
                'id': conv_id,
                'title': result['title'],
                'created': result['created'],
                'message_count': result['message_count'],
                'turn_count': result['turn_count'],
                'source': export_format
            })

            stats.parsed += 1
            stats.total_messages += result['message_count']

        except Exception as e:
            stats.skipped_malformed += 1
            stats.errors.append({'id': conv_id, 'error': str(e)})

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}...")

    # Load existing manifest and merge (keep entries from other formats)
    manifest_path = output_dir / '_manifest.json'
    existing_entries = []
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                existing_manifest = json.load(f)
                # Keep entries from OTHER formats (filter out current format to allow re-parse)
                existing_entries = [
                    e for e in existing_manifest.get('conversations', [])
                    if e.get('source') != export_format
                ]
        except (json.JSONDecodeError, KeyError):
            existing_entries = []

    # Combine entries
    all_entries = existing_entries + manifest_entries

    # Calculate combined date range from all entries
    all_dates = [e['created'][:10] for e in all_entries if e.get('created')]
    if all_dates:
        all_dates.sort()
        date_range = [all_dates[0], all_dates[-1]]
    else:
        date_range = None

    # Write merged manifest
    manifest = {
        'processed_at': datetime.now(tz=timezone.utc).isoformat(),
        'total_conversations': len(all_entries),
        'parsed': len(all_entries),
        'skipped_trivial': stats.skipped_trivial,
        'skipped_malformed': stats.skipped_malformed,
        'date_range': date_range,
        'conversations': all_entries
    }

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 50)
    print("PARSING COMPLETE")
    print("=" * 50)
    print(f"Total conversations:    {stats.total}")
    print(f"Parsed:                 {stats.parsed}")
    print(f"Skipped (trivial <4):   {stats.skipped_trivial}")
    print(f"Skipped (malformed):    {stats.skipped_malformed}")
    print(f"Total messages:         {stats.total_messages}")
    if date_range:
        print(f"Date range:             {date_range[0]} to {date_range[1]}")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"  - {err['id']}: {err['error']}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    print(f"\nOutput: {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
