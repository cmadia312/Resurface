#!/usr/bin/env python3
"""
Core extraction logic for analyzing conversations with LLMs.

Supports both Anthropic and OpenAI APIs.
"""
import json
import re
from config import get_api_key

EXTRACTION_PROMPT = """Analyze this conversation and extract the following. Be specific—include enough context that someone reading this extraction would understand the idea without the original. If a category has nothing notable, omit it.

1. PROJECT_IDEAS: Explicit mentions of something the user wants to build, create, or make. Include the stated or implied motivation.

2. PROBLEMS: Frustrations, inefficiencies, or pain points described. These are implicit project seeds.

3. WORKFLOWS: Systems, processes, or automations the user was trying to create or optimize.

4. TOOLS_EXPLORED: Specific tools, APIs, libraries, or platforms the user was learning or evaluating.

5. QUESTIONS_UNDERLYING: If the user asked variations of the same question, what's the deeper uncertainty or goal?

6. EMOTIONAL_SIGNALS: Note if the user seemed excited, frustrated, obsessive, or stuck. Quote briefly if illustrative.

Return valid JSON matching this schema:
{
  "project_ideas": [{"idea": "...", "motivation": "...", "detail_level": "vague|sketched|detailed"}],
  "problems": [{"problem": "...", "context": "..."}],
  "workflows": [{"workflow": "...", "status": "exploring|building|optimizing"}],
  "tools_explored": ["tool1", "tool2"],
  "underlying_questions": ["..."],
  "emotional_signals": {"tone": "excited|frustrated|curious|stuck|neutral", "notes": "..."}
}

If the conversation contains nothing extractable (purely factual Q&A, casual chat, etc.), return:
{"empty": true, "reason": "..."}"""

RETRY_PROMPT = "Your previous response was not valid JSON. Please return only valid JSON matching the schema."


def format_conversation(conversation: dict) -> str:
    """Format a parsed conversation for the extraction prompt."""
    lines = []
    lines.append(f"Title: {conversation.get('title', 'Untitled')}")
    lines.append(f"Date: {conversation.get('created', 'Unknown')[:10] if conversation.get('created') else 'Unknown'}")
    lines.append("")

    for msg in conversation.get('messages', []):
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')

        # Skip tool messages in the formatted output (they're just markers)
        if role == 'TOOL':
            continue

        lines.append(f"{role}: {content}")
        lines.append("")

    return '\n'.join(lines)


def parse_json_response(text: str) -> dict | None:
    """
    Attempt to parse JSON from LLM response.
    Handles responses that may have markdown code blocks or extra text.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_anthropic(messages: list, config: dict) -> str:
    """Call Anthropic API and return response text."""
    import anthropic

    api_key = get_api_key(config)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=config.get('model', 'claude-sonnet-4-20250514'),
        max_tokens=2048,
        messages=messages
    )

    return response.content[0].text


def call_openai(messages: list, config: dict) -> str:
    """Call OpenAI API and return response text."""
    import openai

    api_key = get_api_key(config)
    client = openai.OpenAI(api_key=api_key)

    # Convert to OpenAI message format
    openai_messages = [{"role": "system", "content": EXTRACTION_PROMPT}]
    for msg in messages:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = client.chat.completions.create(
        model=config.get('model', 'gpt-4o'),
        max_tokens=2048,
        messages=openai_messages
    )

    return response.choices[0].message.content


def call_llm(messages: list, config: dict) -> str:
    """Call the configured LLM provider."""
    provider = config.get('api_provider', 'anthropic')

    if provider == 'anthropic':
        return call_anthropic(messages, config)
    elif provider == 'openai':
        return call_openai(messages, config)
    else:
        raise ValueError(f"Unknown API provider: {provider}")


def extract_conversation(conversation: dict, config: dict) -> dict:
    """
    Extract insights from a parsed conversation using configured LLM.

    Args:
        conversation: Parsed conversation dict (from parser.py output)
        config: Configuration dict

    Returns:
        Extraction result dict, or error dict if extraction fails
    """
    # Format the conversation
    formatted = format_conversation(conversation)

    # Build initial message
    if config.get('api_provider') == 'anthropic':
        messages = [{
            "role": "user",
            "content": f"{EXTRACTION_PROMPT}\n\n---\n\n{formatted}"
        }]
    else:
        # For OpenAI, system prompt is added in call_openai
        messages = [{
            "role": "user",
            "content": formatted
        }]

    retry_attempts = config.get('retry_attempts', 2)

    for attempt in range(retry_attempts + 1):
        try:
            response_text = call_llm(messages, config)
            result = parse_json_response(response_text)

            if result is not None:
                return result

            # JSON parsing failed, add retry message
            if attempt < retry_attempts:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": RETRY_PROMPT})

        except Exception as e:
            # API error
            return {
                "error": True,
                "error_type": "api_error",
                "error_message": str(e)
            }

    # All retries exhausted, return error with raw response
    return {
        "error": True,
        "error_type": "json_parse_error",
        "raw_response": response_text if 'response_text' in dir() else "No response received"
    }


if __name__ == "__main__":
    # Test with a sample conversation
    import sys
    from pathlib import Path
    from config import load_config, validate_config

    config = load_config()
    is_valid, error = validate_config(config)

    if not is_valid:
        print(f"Configuration error: {error}")
        sys.exit(1)

    # Load a sample conversation for testing
    parsed_dir = Path("data/parsed")
    sample_files = list(parsed_dir.glob("*.json"))

    if not sample_files:
        print("No parsed conversations found in data/parsed/")
        sys.exit(1)

    # Skip manifest
    sample_files = [f for f in sample_files if f.name != "_manifest.json"]
    sample_file = sample_files[0]

    print(f"Testing extraction on: {sample_file.name}")

    with open(sample_file, 'r') as f:
        conversation = json.load(f)

    print(f"Title: {conversation.get('title')}")
    print(f"Messages: {conversation.get('message_count')}")
    print("\nExtracting...")

    result = extract_conversation(conversation, config)
    print("\nResult:")
    print(json.dumps(result, indent=2))
