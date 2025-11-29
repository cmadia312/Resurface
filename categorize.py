#!/usr/bin/env python3
"""
Categorization and scoring for Resurface.

Applies multi-dimensional scoring to consolidated ideas:
- Recurrence: Rule-based (count from consolidation)
- Passion: Rule-based (from emotional signals)
- Effort, Monetization, Personal Utility: LLM-assisted

Categories:
- Quick Win: Low effort + High utility + High passion
- Validate: High monetization + Medium effort
- Revive: High recurrence + High passion + Stalled
- Someday: Everything else
"""
import json
import time
import os
import tempfile
from pathlib import Path
from datetime import datetime
from config import load_config, get_api_key

CONSOLIDATED_DIR = Path("data/consolidated")
CONSOLIDATED_FILE = CONSOLIDATED_DIR / "consolidated.json"
CATEGORIZED_FILE = CONSOLIDATED_DIR / "categorized.json"
STATUS_FILE = Path("data/categorization_status.json")


def update_status(message: str, progress_pct: float | None = None,
                  elapsed_sec: float | None = None, eta_sec: float | None = None,
                  complete: bool = False, error: bool = False):
    """Atomically update categorization status file."""
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

# Prompt for LLM-assisted scoring
SCORING_PROMPT = """Score these project ideas on three dimensions. Use 1-5 scale.

For each idea, evaluate:

1. EFFORT (1=weekend project, 2=week, 3=month, 4=few months, 5=year+)
   Consider: technical complexity, number of integrations needed, scope

2. MONETIZATION (1=no revenue potential, 2=small niche, 3=moderate market, 4=solid business, 5=high potential)
   Consider: market size, willingness to pay, competition, business model viability

3. PERSONAL_UTILITY (1=nice to have, 2=occasionally useful, 3=regularly useful, 4=very useful, 5=essential)
   Infer from HOW the user described it. High-utility signals:
   - "I really need this", "would save me hours", "keep running into this problem"
   - Frustration in the motivation
   - Multiple mentions of personal use case
   - Words like "daily", "constantly", "always"

   Low-utility signals:
   - Abstract/theoretical interest
   - "It would be cool if..."
   - No personal use case mentioned

Ideas to score:
{ideas_json}

Return JSON array with this structure for each idea:
[
  {{
    "name": "idea name (exact match from input)",
    "effort": 1-5,
    "monetization": 1-5,
    "personal_utility": 1-5,
    "reasoning": "brief explanation of scores"
  }}
]

Return only valid JSON."""


def load_consolidated() -> dict | None:
    """Load consolidated data."""
    if not CONSOLIDATED_FILE.exists():
        print(f"Consolidated file not found: {CONSOLIDATED_FILE}")
        return None
    with open(CONSOLIDATED_FILE, 'r') as f:
        return json.load(f)


def calculate_recurrence_score(occurrences: int) -> int:
    """
    Rule-based: Score recurrence from occurrence count.
    1 = appeared once
    2 = appeared 2 times
    3 = appeared 3-4 times
    4 = appeared 5-7 times
    5 = appeared 8+ times
    """
    if occurrences <= 1:
        return 1
    elif occurrences == 2:
        return 2
    elif occurrences <= 4:
        return 3
    elif occurrences <= 7:
        return 4
    else:
        return 5


def calculate_passion_score(idea: dict, emotional_timeline: list) -> int:
    """
    Rule-based: Score passion from emotional signals.

    Checks:
    - Tone of source conversations (excited = high, frustrated about problem = medium-high)
    - Evolution notes (more evolution = more engagement)
    - Detail level (more detailed = more invested)
    """
    score = 3  # Default neutral

    source_ids = idea.get('source_ids', [])
    if not source_ids:
        return score

    # Check emotional tones for source conversations
    excited_count = 0
    frustrated_count = 0

    for entry in emotional_timeline:
        if entry.get('conversation_id') in source_ids:
            tone = entry.get('tone', 'neutral')
            if tone == 'excited':
                excited_count += 1
            elif tone == 'frustrated':
                frustrated_count += 1

    # Excited signals passion
    if excited_count >= 2:
        score += 2
    elif excited_count >= 1:
        score += 1

    # Frustrated about a problem can also signal passion (they care enough to be frustrated)
    if frustrated_count >= 1:
        score += 0.5

    # Check detail levels
    detail_levels = idea.get('detail_levels', [])
    detailed_count = sum(1 for d in detail_levels if d == 'detailed')
    if detailed_count >= 2:
        score += 1
    elif detailed_count >= 1:
        score += 0.5

    # Check evolution - if it evolved, user kept thinking about it
    evolution = idea.get('evolution', '')
    if evolution and 'more specific' in evolution.lower():
        score += 0.5
    if evolution and 'matured' in evolution.lower():
        score += 0.5

    # Clamp to 1-5
    return max(1, min(5, int(round(score))))


def get_llm_scores(ideas: list[dict], config: dict) -> dict:
    """
    Call LLM to get effort, monetization, and personal_utility scores.
    Returns dict mapping idea name -> scores dict.
    """
    provider = config.get('api_provider', 'openai')
    api_key = get_api_key(config)

    # Prepare ideas for prompt - extract relevant info
    ideas_for_prompt = []
    for idea in ideas:
        ideas_for_prompt.append({
            "name": idea.get('name', 'Unknown'),
            "description": idea.get('description', ''),
            "motivations": idea.get('motivations', []),
            "occurrences": idea.get('occurrences', 1),
            "evolution": idea.get('evolution', '')
        })

    prompt = SCORING_PROMPT.format(ideas_json=json.dumps(ideas_for_prompt, indent=2))

    print(f"Calling LLM for scoring {len(ideas)} ideas...")

    if provider == 'openai':
        import openai
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=config.get('model', 'gpt-4o-mini'),
            max_tokens=8192,
            messages=[
                {"role": "system", "content": "You are a startup advisor scoring project ideas. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content

    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=config.get('model', 'claude-sonnet-4-20250514'),
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Parse response
    scores_list = parse_json_array(response_text)

    # Convert to dict keyed by name
    scores_dict = {}
    for score in scores_list:
        name = score.get('name', '')
        if name:
            scores_dict[name] = {
                'effort': score.get('effort', 3),
                'monetization': score.get('monetization', 3),
                'personal_utility': score.get('personal_utility', 3),
                'reasoning': score.get('reasoning', '')
            }

    return scores_dict


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

    print("Warning: Could not parse JSON array from LLM response")
    return []


def determine_category(scores: dict) -> str:
    """
    Determine category based on composite scores.

    Categories:
    - quick_win: Low effort + High utility + High passion
    - validate: High monetization + Medium effort
    - revive: High recurrence + High passion + Stalled
    - someday: Everything else
    """
    effort = scores.get('effort', 3)
    monetization = scores.get('monetization', 3)
    utility = scores.get('personal_utility', 3)
    passion = scores.get('passion', 3)
    recurrence = scores.get('recurrence', 1)

    # Quick win: Easy to build, useful to you, you're excited about it
    if effort <= 2 and utility >= 4 and passion >= 3:
        return 'quick_win'

    # Validate: Could make money, reasonable effort
    if monetization >= 4 and effort <= 4:
        return 'validate'

    # Revive: Kept coming back to it, were passionate, but seems stalled
    if recurrence >= 3 and passion >= 3:
        return 'revive'

    return 'someday'


def calculate_composite_score(scores: dict) -> int:
    """Calculate composite score (sum of all dimensions)."""
    return (
        scores.get('effort', 3) +
        scores.get('monetization', 3) +
        scores.get('personal_utility', 3) +
        scores.get('passion', 3) +
        scores.get('recurrence', 1)
    )


def run_categorization() -> dict:
    """
    Run full categorization pipeline.

    1. Load consolidated data
    2. Calculate rule-based scores (recurrence, passion)
    3. Get LLM scores (effort, monetization, utility)
    4. Determine categories
    5. Save results
    """
    def calc_eta(elapsed: float, progress: float) -> float | None:
        """Calculate ETA based on elapsed time and progress percentage."""
        if progress <= 0:
            return None
        total_estimated = elapsed / (progress / 100)
        return total_estimated - elapsed

    try:
        start_time = time.time()
        update_status("Starting categorization...", progress_pct=0, elapsed_sec=0, eta_sec=None)
        config = load_config()

        print("Loading consolidated data...")
        elapsed = time.time() - start_time
        update_status("Loading consolidated data...", progress_pct=5, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 5))
        data = load_consolidated()
        if not data:
            print("No consolidated data found. Run consolidation first.")
            update_status("No consolidated data found. Run consolidation first.", elapsed_sec=time.time() - start_time, complete=True)
            return {}

        ideas = data.get('idea_clusters', [])
        emotional_timeline = data.get('emotional_timeline', [])

        if not ideas:
            print("No ideas to categorize.")
            update_status("No ideas to categorize.", elapsed_sec=time.time() - start_time, complete=True)
            return {}

        print(f"Categorizing {len(ideas)} ideas...")
        elapsed = time.time() - start_time
        update_status(f"Found {len(ideas)} ideas to categorize", progress_pct=10, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 10))

        # Calculate rule-based scores
        print("\nCalculating rule-based scores (recurrence, passion)...")
        elapsed = time.time() - start_time
        update_status("Calculating rule-based scores...", progress_pct=20, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 20))
        for idea in ideas:
            idea['scores'] = {
                'recurrence': calculate_recurrence_score(idea.get('occurrences', 1)),
                'passion': calculate_passion_score(idea, emotional_timeline)
            }

        # Get LLM scores
        print("\nGetting LLM scores (effort, monetization, utility)...")
        elapsed = time.time() - start_time
        update_status(f"Getting LLM scores for {len(ideas)} ideas...", progress_pct=30, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 30))
        llm_scores = get_llm_scores(ideas, config)

        # Respect rate limit
        rate_limit = config.get('requests_per_minute', 60)
        delay = 60.0 / rate_limit
        time.sleep(delay)

        # Merge scores and determine categories
        print("\nMerging scores and determining categories...")
        elapsed = time.time() - start_time
        update_status("Merging scores and determining categories...", progress_pct=80, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 80))
        categorized_ideas = []

        for idea in ideas:
            name = idea.get('name', 'Unknown')

            # Get LLM scores if available
            if name in llm_scores:
                idea['scores']['effort'] = llm_scores[name].get('effort', 3)
                idea['scores']['monetization'] = llm_scores[name].get('monetization', 3)
                idea['scores']['personal_utility'] = llm_scores[name].get('personal_utility', 3)
                idea['scores']['llm_reasoning'] = llm_scores[name].get('reasoning', '')
            else:
                # Default scores if LLM didn't return for this idea
                idea['scores']['effort'] = 3
                idea['scores']['monetization'] = 3
                idea['scores']['personal_utility'] = 3
                idea['scores']['llm_reasoning'] = 'No LLM scoring available'

            # Calculate composite and category
            idea['composite_score'] = calculate_composite_score(idea['scores'])
            idea['category'] = determine_category(idea['scores'])

            categorized_ideas.append(idea)

        # Sort by composite score descending
        categorized_ideas.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        elapsed = time.time() - start_time
        update_status("Sorting and saving results...", progress_pct=90, elapsed_sec=elapsed, eta_sec=calc_eta(elapsed, 90))

        # Create categorized output
        result = {
            'ideas': categorized_ideas,
            'by_category': {
                'quick_win': [i for i in categorized_ideas if i['category'] == 'quick_win'],
                'validate': [i for i in categorized_ideas if i['category'] == 'validate'],
                'revive': [i for i in categorized_ideas if i['category'] == 'revive'],
                'someday': [i for i in categorized_ideas if i['category'] == 'someday']
            },
            'metadata': {
                'categorized_at': datetime.now().isoformat(),
                'total_ideas': len(categorized_ideas),
                'category_counts': {
                    'quick_win': len([i for i in categorized_ideas if i['category'] == 'quick_win']),
                    'validate': len([i for i in categorized_ideas if i['category'] == 'validate']),
                    'revive': len([i for i in categorized_ideas if i['category'] == 'revive']),
                    'someday': len([i for i in categorized_ideas if i['category'] == 'someday'])
                }
            }
        }

        # Save
        with open(CATEGORIZED_FILE, 'w') as f:
            json.dump(result, f, indent=2)

        final_elapsed = time.time() - start_time
        print(f"\n=== Categorization Complete ===")
        print(f"Total ideas: {len(categorized_ideas)}")
        print(f"Quick Wins: {result['metadata']['category_counts']['quick_win']}")
        print(f"Validate: {result['metadata']['category_counts']['validate']}")
        print(f"Revive: {result['metadata']['category_counts']['revive']}")
        print(f"Someday: {result['metadata']['category_counts']['someday']}")
        print(f"Total time: {final_elapsed:.1f}s")
        print(f"\nSaved to: {CATEGORIZED_FILE}")

        counts = result['metadata']['category_counts']
        update_status(
            f"Complete: {counts['quick_win']} quick wins, {counts['validate']} validate, "
            f"{counts['revive']} revive, {counts['someday']} someday",
            progress_pct=100, elapsed_sec=final_elapsed, eta_sec=0, complete=True
        )

        return result

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        update_status(error_msg, error=True)
        raise


if __name__ == "__main__":
    run_categorization()
