#!/usr/bin/env python3
"""
Creative Synthesis Engine for Resurface.

Generates novel project ideas by analyzing patterns in user's conversation history.
Uses 4 strategies:
1. Passion Intersections - Combine top themes
2. Problem-Solution - Match problems with tools
3. Profile-Based - Generate from holistic user profile
4. Time Capsule - Resurface old ideas with new context
"""
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from config import load_config, get_api_key

# Paths
CONSOLIDATED_DIR = Path("data/consolidated")
CONSOLIDATED_FILE = CONSOLIDATED_DIR / "consolidated.json"
CATEGORIZED_FILE = CONSOLIDATED_DIR / "categorized.json"
SYNTHESIZED_DIR = Path("data/synthesized")
PROFILE_FILE = SYNTHESIZED_DIR / "passion_profile.json"
GENERATED_FILE = SYNTHESIZED_DIR / "generated_ideas.json"
SAVED_FILE = SYNTHESIZED_DIR / "saved_ideas.json"
DEVELOPED_DIR = SYNTHESIZED_DIR / "developed"
STATUS_FILE = Path("data/synthesis_status.json")


def update_status(message: str, progress_pct: float = None,
                  complete: bool = False, error: bool = False):
    """Atomically update synthesis status file for UI polling."""
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
        os.replace(tmp_path, STATUS_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def parse_json_response(text: str) -> dict | list | None:
    """Parse JSON from LLM response, handling markdown blocks."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object/array in text
    json_match = re.search(r'[\[{][\s\S]*[\]}]', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_llm(prompt: str, config: dict, system_prompt: str = None) -> str:
    """Call configured LLM provider."""
    provider = config.get('api_provider', 'openai')
    api_key = get_api_key(config)

    if provider == 'openai':
        import openai
        client = openai.OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=config.get('model', 'gpt-4o-mini'),
            max_tokens=8192,
            messages=messages
        )
        return response.choices[0].message.content

    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = client.messages.create(
            model=config.get('model', 'claude-sonnet-4-20250514'),
            max_tokens=8192,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unknown API provider: {provider}")


def load_synthesis_inputs() -> tuple[dict, dict] | None:
    """
    Load consolidated and categorized data.

    Returns:
        Tuple of (consolidated_data, categorized_data) or None if not found
    """
    if not CONSOLIDATED_FILE.exists():
        print(f"Consolidated file not found: {CONSOLIDATED_FILE}")
        return None

    if not CATEGORIZED_FILE.exists():
        print(f"Categorized file not found: {CATEGORIZED_FILE}")
        return None

    with open(CONSOLIDATED_FILE, 'r') as f:
        consolidated = json.load(f)

    with open(CATEGORIZED_FILE, 'r') as f:
        categorized = json.load(f)

    return consolidated, categorized


def prepare_profile_data_for_llm(consolidated: dict, categorized: dict,
                                  max_ideas: int = 30, max_problems: int = 20,
                                  max_workflows: int = 15, max_emotions: int = 50) -> dict:
    """Prepare truncated data for LLM to avoid context length issues."""
    # Get top ideas by composite score (already sorted)
    ideas = categorized.get("ideas", [])[:max_ideas]
    slim_ideas = [{
        "name": i.get("name"),
        "description": i.get("description", "")[:200],
        "category": i.get("category"),
        "occurrences": i.get("occurrences"),
        "scores": {k: v for k, v in i.get("scores", {}).items() if k in ["passion", "recurrence"]}
    } for i in ideas]

    # Top problems
    problems = consolidated.get("problem_clusters", [])[:max_problems]
    slim_problems = [{"name": p.get("name"), "description": p.get("description", "")[:200]} for p in problems]

    # Top workflows
    workflows = consolidated.get("workflow_clusters", [])[:max_workflows]
    slim_workflows = [{"name": w.get("name"), "description": w.get("description", "")[:200]} for w in workflows]

    # Tool frequency - top 20
    tools = consolidated.get("tool_frequency", {})
    top_tools = dict(sorted(tools.items(), key=lambda x: x[1], reverse=True)[:20])

    # Emotional timeline - sample evenly
    emotions = consolidated.get("emotional_timeline", [])
    if len(emotions) > max_emotions:
        step = len(emotions) // max_emotions
        emotions = emotions[::step][:max_emotions]

    return {
        "top_ideas": slim_ideas,
        "top_problems": slim_problems,
        "top_workflows": slim_workflows,
        "tool_frequency": top_tools,
        "emotional_samples": emotions,
        "total_counts": {
            "ideas": len(categorized.get("ideas", [])),
            "problems": len(consolidated.get("problem_clusters", [])),
            "workflows": len(consolidated.get("workflow_clusters", []))
        }
    }


def build_passion_profile(consolidated: dict, categorized: dict, config: dict) -> dict:
    """
    Build comprehensive passion profile from all available data.
    Uses LLM to synthesize patterns.
    """
    # Prepare truncated data for the LLM to avoid context length issues
    profile_data = prepare_profile_data_for_llm(consolidated, categorized)

    schema = """{
  "core_themes": [{"theme": "string", "strength": 1-5, "evidence": "string"}],
  "tool_expertise": [{"tool": "string", "mentions": number, "contexts": ["string"]}],
  "recurring_problems": [{"problem": "string", "frequency": number, "impact": "string"}],
  "emotional_patterns": {
    "excited_about": ["string"],
    "frustrated_by": ["string"],
    "curious_about": ["string"]
  },
  "underlying_questions": ["string"],
  "high_passion_ideas": [{"name": "string", "passion_score": 1-5}],
  "summary": "2-3 sentence description"
}"""

    total = profile_data.get("total_counts", {})
    prompt = f"""Analyze this user's conversation history data and build a passion profile.

DATA PROVIDED (sampled from {total.get('ideas', 0)} ideas, {total.get('problems', 0)} problems, {total.get('workflows', 0)} workflows):
- Top Ideas (highest scoring project ideas with categories and passion/recurrence scores)
- Top Problems (pain points)
- Top Workflows (automations and processes)
- Tool Frequency (top technologies mentioned)
- Emotional Samples (conversation tones over time)

{json.dumps(profile_data, indent=2)}

Synthesize this into a passion profile with:
1. core_themes: Top 5-10 recurring themes/interests (with evidence from the data)
2. tool_expertise: Tools they use/explore most (with contexts)
3. recurring_problems: Pain points that keep appearing
4. emotional_patterns: What excites them vs frustrates them vs makes them curious
5. underlying_questions: Deeper uncertainties they keep exploring
6. high_passion_ideas: Ideas with highest passion scores
7. summary: 2-3 sentence description of this person's interests/focus

Return JSON matching this schema:
{schema}"""

    system_prompt = "You are an expert at analyzing patterns in human interests and synthesizing insights. Return only valid JSON."

    response = call_llm(prompt, config, system_prompt)
    profile = parse_json_response(response)

    if profile is None:
        print("Warning: Could not parse passion profile from LLM response")
        profile = {
            "core_themes": [],
            "tool_expertise": [],
            "recurring_problems": [],
            "emotional_patterns": {"excited_about": [], "frustrated_by": [], "curious_about": []},
            "underlying_questions": [],
            "high_passion_ideas": [],
            "summary": "Profile generation failed"
        }

    # Add metadata
    profile["date_range"] = {
        "start": consolidated.get("emotional_timeline", [{}])[0].get("date", "unknown") if consolidated.get("emotional_timeline") else "unknown",
        "end": consolidated.get("emotional_timeline", [{}])[-1].get("date", "unknown") if consolidated.get("emotional_timeline") else "unknown"
    }
    profile["generated_at"] = datetime.now().isoformat()

    return profile


def extract_themes_for_intersection(profile: dict) -> list[str]:
    """Extract top themes suitable for intersection generation."""
    themes = []
    for theme_obj in profile.get("core_themes", []):
        if isinstance(theme_obj, dict):
            themes.append(theme_obj.get("theme", ""))
        elif isinstance(theme_obj, str):
            themes.append(theme_obj)

    # Return top 8 themes
    return [t for t in themes if t][:8]


def extract_top_tools(profile: dict, consolidated: dict) -> list[str]:
    """Extract top tools from profile and consolidated data."""
    tools = []

    # From profile
    for tool_obj in profile.get("tool_expertise", []):
        if isinstance(tool_obj, dict):
            tools.append(tool_obj.get("tool", ""))
        elif isinstance(tool_obj, str):
            tools.append(tool_obj)

    # From consolidated tool_frequency
    tool_freq = consolidated.get("tool_frequency", {})
    sorted_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)
    for tool, _ in sorted_tools[:10]:
        if tool not in tools:
            tools.append(tool)

    return [t for t in tools if t][:15]


def generate_intersection_ideas(themes: list[str], tools: list[str],
                                 profile: dict, config: dict) -> list[dict]:
    """Strategy A: Generate ideas at intersection of 2-3 themes."""

    prompt = f"""You are helping someone discover project ideas at the intersection of their passions.

USER PROFILE SUMMARY:
{profile.get('summary', 'A developer exploring various interests.')}

THEIR TOP THEMES:
{json.dumps(themes, indent=2)}

TOOLS THEY KNOW:
{json.dumps(tools, indent=2)}

Generate 5 NOVEL project ideas that creatively combine 2-3 of these themes.
These should be ideas the user has NOT explicitly mentioned - synthesize something new.

For each idea provide:
- name: Catchy, memorable project name
- description: 2-3 sentences explaining what it does
- themes_combined: Which themes this intersects
- tools_suggested: Relevant tools from their expertise
- why_exciting: Why this person specifically would love building this

Return JSON array:
[{{"name": "...", "description": "...", "themes_combined": [...], "tools_suggested": [...], "why_exciting": "..."}}]"""

    system_prompt = "You are a creative project ideation expert. Generate novel, exciting project ideas. Return only valid JSON array."

    response = call_llm(prompt, config, system_prompt)
    ideas = parse_json_response(response)

    if ideas is None or not isinstance(ideas, list):
        print("Warning: Could not parse intersection ideas from LLM response")
        return []

    # Add strategy tag
    for idea in ideas:
        idea["strategy"] = "intersection"
        idea["id"] = f"gen_{uuid.uuid4().hex[:8]}"

    return ideas


def generate_solution_ideas(problems: list[dict], tools: list[str],
                            profile: dict, config: dict) -> list[dict]:
    """Strategy B: Generate solutions to recurring problems using known tools."""

    prompt = f"""You are helping someone solve their recurring problems using tools they already know.

USER'S RECURRING PROBLEMS:
{json.dumps(problems, indent=2)}

TOOLS THEY'RE FAMILIAR WITH:
{json.dumps(tools, indent=2)}

USER PROFILE SUMMARY:
{profile.get('summary', 'A developer exploring various interests.')}

Generate 5 practical project ideas that solve these problems using their toolkit.
Focus on ACTIONABLE solutions they could realistically build.

For each idea provide:
- name: Clear, descriptive project name
- description: What it does and how it solves the problem
- problem_addressed: Which problem(s) this solves
- tools_used: Which of their known tools this uses
- why_practical: Why this is achievable for them

Return JSON array:
[{{"name": "...", "description": "...", "problem_addressed": "...", "tools_used": [...], "why_practical": "..."}}]"""

    system_prompt = "You are a practical problem-solving expert. Generate actionable, buildable solutions. Return only valid JSON array."

    response = call_llm(prompt, config, system_prompt)
    ideas = parse_json_response(response)

    if ideas is None or not isinstance(ideas, list):
        print("Warning: Could not parse solution ideas from LLM response")
        return []

    for idea in ideas:
        idea["strategy"] = "problem_solution"
        idea["id"] = f"gen_{uuid.uuid4().hex[:8]}"

    return ideas


def generate_profile_ideas(profile: dict, config: dict) -> list[dict]:
    """Strategy D: Generate ideas based on holistic profile understanding."""

    prompt = f"""You are roleplaying as someone with the following passion profile:

{json.dumps(profile, indent=2)}

Based on this person's interests, frustrations, tools, and patterns:

Generate 5 project ideas that would EXCITE this person.
These should feel like "why didn't I think of that?" moments.
They should NOT be ideas they've already mentioned - synthesize something NEW.

For each idea:
- name: Project name
- description: What it does (2-3 sentences)
- profile_alignment: How this matches their interests/needs
- novelty_factor: Why this is a fresh angle they haven't considered

Return JSON array:
[{{"name": "...", "description": "...", "profile_alignment": "...", "novelty_factor": "..."}}]"""

    system_prompt = "You are an empathetic ideation expert who deeply understands human interests. Generate surprising, delightful ideas. Return only valid JSON array."

    response = call_llm(prompt, config, system_prompt)
    ideas = parse_json_response(response)

    if ideas is None or not isinstance(ideas, list):
        print("Warning: Could not parse profile ideas from LLM response")
        return []

    for idea in ideas:
        idea["strategy"] = "profile_based"
        idea["id"] = f"gen_{uuid.uuid4().hex[:8]}"

    return ideas


def generate_time_capsule_ideas(categorized: dict, consolidated: dict,
                                 config: dict) -> list[dict]:
    """Strategy F: Resurface old high-passion ideas with updated context."""

    ideas = categorized.get("ideas", [])
    tool_freq = consolidated.get("tool_frequency", {})

    # Find ideas that are old (check date_range) and have high passion
    old_high_passion = []
    current_date = datetime.now()

    for idea in ideas:
        date_range = idea.get("date_range", [])
        passion_score = idea.get("scores", {}).get("passion", 0)

        if date_range and len(date_range) > 0 and passion_score >= 4:
            try:
                first_date = datetime.fromisoformat(date_range[0].replace('Z', '+00:00').split('T')[0])
                months_ago = (current_date - first_date).days / 30
                if months_ago >= 1:  # At least 1 month old (relaxed for testing)
                    old_high_passion.append({
                        "idea": idea,
                        "months_ago": int(months_ago),
                        "first_date": date_range[0]
                    })
            except (ValueError, TypeError):
                continue

    # Sort by months_ago descending, take top 5
    old_high_passion.sort(key=lambda x: x["months_ago"], reverse=True)
    old_high_passion = old_high_passion[:5]

    if not old_high_passion:
        print("No old high-passion ideas found for time capsule")
        return []

    time_capsule_ideas = []
    all_tools = list(tool_freq.keys())

    for item in old_high_passion:
        idea = item["idea"]
        months_ago = item["months_ago"]

        prompt = f"""This user had an idea {months_ago} months ago that they were excited about:

ORIGINAL IDEA:
Name: {idea.get('name', 'Unknown')}
Description: {idea.get('description', '')}
Motivation: {json.dumps(idea.get('motivations', []))}
Passion Score: {idea.get('scores', {}).get('passion', 0)}/5
Date First Mentioned: {item['first_date']}

TOOLS THEY'VE EXPLORED:
{json.dumps(all_tools[:20], indent=2)}

Write two things:

1. A brief "letter from past self" (2-3 sentences) - what past-them would say about why this mattered

2. An UPDATED VERSION of this idea that could incorporate their knowledge/tools

Return JSON:
{{
  "letter_from_past": "...",
  "updated_name": "...",
  "updated_vision": "How to approach this now with new knowledge",
  "tools_to_use": ["..."]
}}"""

        system_prompt = "You are helping someone reconnect with their past ideas. Be warm and encouraging. Return only valid JSON."

        try:
            response = call_llm(prompt, config, system_prompt)
            result = parse_json_response(response)

            if result:
                time_capsule_ideas.append({
                    "id": f"gen_{uuid.uuid4().hex[:8]}",
                    "name": result.get("updated_name", idea.get("name", "Unknown")),
                    "description": result.get("updated_vision", ""),
                    "original_idea": idea.get("name", "Unknown"),
                    "original_date": item["first_date"],
                    "months_ago": months_ago,
                    "original_passion": idea.get("scores", {}).get("passion", 0),
                    "letter_from_past": result.get("letter_from_past", ""),
                    "tools_suggested": result.get("tools_to_use", []),
                    "strategy": "time_capsule"
                })
        except Exception as e:
            print(f"Error generating time capsule for {idea.get('name')}: {e}")
            continue

        # Rate limiting
        time.sleep(1)

    return time_capsule_ideas


def deduplicate_generated_ideas(ideas: list[dict], config: dict) -> list[dict]:
    """Use LLM to identify and merge semantically similar generated ideas."""

    if len(ideas) <= 3:
        return ideas

    # Prepare simplified list for LLM
    ideas_for_dedup = []
    for idea in ideas:
        ideas_for_dedup.append({
            "id": idea.get("id"),
            "name": idea.get("name"),
            "description": idea.get("description", "")[:200],
            "strategy": idea.get("strategy")
        })

    prompt = f"""Review these generated project ideas and identify any that are essentially the same idea.

IDEAS:
{json.dumps(ideas_for_dedup, indent=2)}

For each group of similar ideas, pick the best name and description, and merge them.
Return the IDs to keep and which IDs were merged into them.

Return JSON:
{{
  "keep": [
    {{"id": "id_to_keep", "merged_ids": ["id1", "id2"]}}
  ],
  "unique": ["id3", "id4"]
}}

If all ideas are unique, return:
{{"keep": [], "unique": ["all", "the", "ids"]}}"""

    system_prompt = "You are an expert at identifying semantic similarity. Return only valid JSON."

    response = call_llm(prompt, config, system_prompt)
    result = parse_json_response(response)

    if result is None:
        print("Warning: Could not parse deduplication result, returning all ideas")
        return ideas

    # Build set of IDs to keep
    keep_ids = set()
    merged_map = {}  # id -> list of merged ids

    for item in result.get("keep", []):
        keep_id = item.get("id")
        if keep_id:
            keep_ids.add(keep_id)
            merged_map[keep_id] = item.get("merged_ids", [])

    for unique_id in result.get("unique", []):
        keep_ids.add(unique_id)

    # Filter ideas
    deduped = []
    id_to_idea = {idea.get("id"): idea for idea in ideas}

    for idea in ideas:
        idea_id = idea.get("id")
        if idea_id in keep_ids:
            # Add merged_from if applicable
            if idea_id in merged_map and merged_map[idea_id]:
                merged_strategies = set([idea.get("strategy")])
                for merged_id in merged_map[idea_id]:
                    if merged_id in id_to_idea:
                        merged_strategies.add(id_to_idea[merged_id].get("strategy"))
                idea["merged_from_strategies"] = list(merged_strategies)
            deduped.append(idea)

    # If dedup failed, return originals
    if not deduped:
        return ideas

    return deduped


def score_generated_ideas(ideas: list[dict], profile: dict, config: dict) -> list[dict]:
    """Score all generated ideas on 5 dimensions."""

    if not ideas:
        return ideas

    ideas_for_scoring = []
    for idea in ideas:
        ideas_for_scoring.append({
            "id": idea.get("id"),
            "name": idea.get("name"),
            "description": idea.get("description", "")[:300],
            "strategy": idea.get("strategy")
        })

    prompt = f"""Score these generated project ideas on 5 dimensions (1-5 scale each):

IDEAS:
{json.dumps(ideas_for_scoring, indent=2)}

USER PROFILE SUMMARY:
{profile.get('summary', 'A developer with various interests.')}

For each idea, score:
1. effort: 1=weekend, 2=week, 3=month, 4=few months, 5=year+
2. monetization: 1=none, 2=small niche, 3=moderate, 4=solid business, 5=high potential
3. personal_utility: 1=nice to have, 3=regularly useful, 5=essential
4. passion_alignment: 1=low match, 3=moderate, 5=perfect match to their interests
5. novelty: 1=common idea, 3=somewhat fresh, 5=highly original

Return JSON array:
[{{"id": "...", "effort": N, "monetization": N, "personal_utility": N, "passion_alignment": N, "novelty": N}}]"""

    system_prompt = "You are an expert at evaluating project ideas. Return only valid JSON array."

    response = call_llm(prompt, config, system_prompt)
    scores_list = parse_json_response(response)

    if scores_list is None or not isinstance(scores_list, list):
        print("Warning: Could not parse scores, using defaults")
        for idea in ideas:
            idea["scores"] = {
                "effort": 3, "monetization": 3, "personal_utility": 3,
                "passion_alignment": 3, "novelty": 3
            }
            idea["composite_score"] = 15
        return ideas

    # Map scores back to ideas
    scores_map = {s.get("id"): s for s in scores_list}

    for idea in ideas:
        idea_id = idea.get("id")
        if idea_id in scores_map:
            s = scores_map[idea_id]
            idea["scores"] = {
                "effort": s.get("effort", 3),
                "monetization": s.get("monetization", 3),
                "personal_utility": s.get("personal_utility", 3),
                "passion_alignment": s.get("passion_alignment", 3),
                "novelty": s.get("novelty", 3)
            }
        else:
            idea["scores"] = {
                "effort": 3, "monetization": 3, "personal_utility": 3,
                "passion_alignment": 3, "novelty": 3
            }

        idea["composite_score"] = sum(idea["scores"].values())

    # Sort by composite score descending
    ideas.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    return ideas


def run_synthesis() -> dict:
    """
    Main synthesis pipeline.
    """
    start_time = time.time()

    try:
        update_status("Loading data...", progress_pct=5)

        # Load inputs
        inputs = load_synthesis_inputs()
        if inputs is None:
            update_status("Error: Could not load consolidated/categorized data", error=True)
            return {}

        consolidated, categorized = inputs
        config = load_config()

        # Ensure output directories exist
        SYNTHESIZED_DIR.mkdir(parents=True, exist_ok=True)
        DEVELOPED_DIR.mkdir(parents=True, exist_ok=True)

        # Build passion profile
        update_status("Building passion profile...", progress_pct=15)
        profile = build_passion_profile(consolidated, categorized, config)

        # Save profile
        with open(PROFILE_FILE, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"Saved passion profile to {PROFILE_FILE}")

        # Extract themes and tools for generation
        themes = extract_themes_for_intersection(profile)
        tools = extract_top_tools(profile, consolidated)
        problems = profile.get("recurring_problems", [])

        all_generated = []

        # Strategy A: Intersections
        update_status("Generating intersection ideas...", progress_pct=30)
        intersection_ideas = generate_intersection_ideas(themes, tools, profile, config)
        all_generated.extend(intersection_ideas)
        print(f"Generated {len(intersection_ideas)} intersection ideas")
        time.sleep(1)  # Rate limiting

        # Strategy B: Problem-Solutions
        update_status("Generating problem-solution ideas...", progress_pct=45)
        solution_ideas = generate_solution_ideas(problems, tools, profile, config)
        all_generated.extend(solution_ideas)
        print(f"Generated {len(solution_ideas)} solution ideas")
        time.sleep(1)

        # Strategy D: Profile-based
        update_status("Generating profile-based ideas...", progress_pct=60)
        profile_ideas = generate_profile_ideas(profile, config)
        all_generated.extend(profile_ideas)
        print(f"Generated {len(profile_ideas)} profile-based ideas")
        time.sleep(1)

        # Strategy F: Time Capsules
        update_status("Generating time capsule ideas...", progress_pct=70)
        capsule_ideas = generate_time_capsule_ideas(categorized, consolidated, config)
        all_generated.extend(capsule_ideas)
        print(f"Generated {len(capsule_ideas)} time capsule ideas")

        # Deduplicate
        update_status("Deduplicating ideas...", progress_pct=80)
        deduped_ideas = deduplicate_generated_ideas(all_generated, config)
        print(f"After deduplication: {len(deduped_ideas)} ideas")
        time.sleep(1)

        # Score
        update_status("Scoring ideas...", progress_pct=90)
        scored_ideas = score_generated_ideas(deduped_ideas, profile, config)

        # Organize by strategy
        by_strategy = {
            "intersection": [],
            "problem_solution": [],
            "profile_based": [],
            "time_capsule": []
        }

        for idea in scored_ideas:
            strategy = idea.get("strategy", "unknown")
            if strategy in by_strategy:
                by_strategy[strategy].append(idea.get("id"))

        # Build result
        result = {
            "ideas": scored_ideas,
            "by_strategy": by_strategy,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_generated": len(all_generated),
                "after_dedup": len(scored_ideas),
                "by_strategy_counts": {k: len(v) for k, v in by_strategy.items()},
                "profile_version": profile.get("generated_at")
            }
        }

        # Save results
        with open(GENERATED_FILE, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved generated ideas to {GENERATED_FILE}")

        elapsed = time.time() - start_time
        update_status(
            f"Complete! Generated {len(scored_ideas)} ideas in {elapsed:.1f}s",
            progress_pct=100, complete=True
        )

        return {
            "profile": profile,
            "generated_ideas": result,
            "all_ideas": scored_ideas
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        update_status(error_msg, error=True)
        return {}


def save_idea(idea_id: str) -> bool:
    """Save an idea to saved_ideas.json."""
    if not GENERATED_FILE.exists():
        return False

    with open(GENERATED_FILE, 'r') as f:
        generated = json.load(f)

    # Find the idea
    idea = None
    for i in generated.get("ideas", []):
        if i.get("id") == idea_id:
            idea = i.copy()
            break

    if idea is None:
        return False

    idea["saved_at"] = datetime.now().isoformat()
    idea["status"] = "saved"

    # Load or create saved file
    if SAVED_FILE.exists():
        with open(SAVED_FILE, 'r') as f:
            saved = json.load(f)
    else:
        saved = {"ideas": []}

    # Check if already saved
    existing_ids = {i.get("id") for i in saved.get("ideas", [])}
    if idea_id not in existing_ids:
        saved["ideas"].append(idea)
        with open(SAVED_FILE, 'w') as f:
            json.dump(saved, f, indent=2)

    return True


def dismiss_idea(idea_id: str) -> bool:
    """Mark an idea as dismissed."""
    if not GENERATED_FILE.exists():
        return False

    with open(GENERATED_FILE, 'r') as f:
        generated = json.load(f)

    # Update status
    for idea in generated.get("ideas", []):
        if idea.get("id") == idea_id:
            idea["status"] = "dismissed"
            break

    with open(GENERATED_FILE, 'w') as f:
        json.dump(generated, f, indent=2)

    return True


def develop_idea_further(idea_id: str) -> dict | None:
    """Generate detailed project specification for an idea."""
    config = load_config()

    # Find the idea
    idea = None

    if SAVED_FILE.exists():
        with open(SAVED_FILE, 'r') as f:
            saved = json.load(f)
        for i in saved.get("ideas", []):
            if i.get("id") == idea_id:
                idea = i
                break

    if idea is None and GENERATED_FILE.exists():
        with open(GENERATED_FILE, 'r') as f:
            generated = json.load(f)
        for i in generated.get("ideas", []):
            if i.get("id") == idea_id:
                idea = i
                break

    if idea is None:
        return None

    # Load profile for context
    profile_summary = "A developer with various interests."
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, 'r') as f:
            profile = json.load(f)
        profile_summary = profile.get("summary", profile_summary)

    prompt = f"""Develop a detailed project specification for this idea:

IDEA:
{json.dumps(idea, indent=2)}

USER PROFILE (for context):
{profile_summary}

Generate a comprehensive project plan:

1. Full Description (300-500 words): Detailed explanation of what this project does, who it's for, and why it matters

2. Tech Stack: Recommended technologies with rationale

3. MVP Scope: 3-5 features for minimum viable product

4. Challenges: 3-4 potential challenges with mitigations

5. First 5 Steps: Concrete actions to start today

6. Effort Estimate: weekend/week/month/quarter

7. Connections: How this relates to user's other interests

Return JSON:
{{
  "full_description": "...",
  "tech_stack": {{"core": "...", "storage": "...", "ui": "...", "rationale": "..."}},
  "mvp_scope": ["feature1", "feature2", ...],
  "challenges": [{{"challenge": "...", "mitigation": "..."}}],
  "first_steps": ["step1", "step2", "step3", "step4", "step5"],
  "effort_estimate": "week",
  "connections_to_interests": "..."
}}"""

    system_prompt = "You are an expert project planner. Create actionable, realistic project specifications. Return only valid JSON."

    response = call_llm(prompt, config, system_prompt)
    spec = parse_json_response(response)

    if spec is None:
        return None

    spec["idea_id"] = idea_id
    spec["idea_name"] = idea.get("name", "Unknown")
    spec["developed_at"] = datetime.now().isoformat()

    # Save to developed directory
    spec_file = DEVELOPED_DIR / f"{idea_id}.json"
    with open(spec_file, 'w') as f:
        json.dump(spec, f, indent=2)

    return spec


def get_synthesis_status() -> dict:
    """Get current synthesis status."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {"message": "Not started", "progress": 0, "complete": False}


def load_generated_ideas() -> dict:
    """Load generated ideas for UI display."""
    if GENERATED_FILE.exists():
        with open(GENERATED_FILE, 'r') as f:
            return json.load(f)
    return {"ideas": [], "by_strategy": {}, "metadata": {}}


def load_passion_profile() -> dict:
    """Load passion profile for UI display."""
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_saved_ideas() -> list[dict]:
    """Load saved ideas."""
    if SAVED_FILE.exists():
        with open(SAVED_FILE, 'r') as f:
            data = json.load(f)
        return data.get("ideas", [])
    return []


def get_developed_ideas() -> list[dict]:
    """
    List all developed specifications from data/synthesized/developed/.

    Returns:
        List of dicts with idea_id, idea_name, developed_at for each spec
    """
    if not DEVELOPED_DIR.exists():
        return []

    developed = []
    for spec_file in DEVELOPED_DIR.glob("*.json"):
        try:
            with open(spec_file, 'r') as f:
                spec = json.load(f)
            developed.append({
                "idea_id": spec.get("idea_id", spec_file.stem),
                "idea_name": spec.get("idea_name", "Unknown"),
                "developed_at": spec.get("developed_at", "Unknown"),
                "effort_estimate": spec.get("effort_estimate", "Unknown")
            })
        except Exception:
            continue

    # Sort by developed_at descending (most recent first)
    developed.sort(key=lambda x: x.get("developed_at", ""), reverse=True)
    return developed


def get_developed_spec(idea_id: str) -> dict | None:
    """
    Load a specific developed specification.

    Args:
        idea_id: The ID of the idea to load

    Returns:
        The full specification dict, or None if not found
    """
    spec_file = DEVELOPED_DIR / f"{idea_id}.json"
    if spec_file.exists():
        with open(spec_file, 'r') as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print("Running synthesis pipeline...")
    result = run_synthesis()

    if result:
        print("\n=== Synthesis Complete ===")
        print(f"Profile summary: {result.get('profile', {}).get('summary', 'N/A')}")
        print(f"Total ideas generated: {len(result.get('all_ideas', []))}")

        for idea in result.get('all_ideas', [])[:5]:
            print(f"\n- {idea.get('name')} ({idea.get('strategy')})")
            print(f"  Score: {idea.get('composite_score', 0)}")
            print(f"  {idea.get('description', '')[:100]}...")
    else:
        print("Synthesis failed. Check data/synthesis_status.json for details.")
