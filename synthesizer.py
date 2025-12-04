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

from config import load_config
from llm_provider import call_llm as call_llm_provider, make_array_schema
from schemas import (
    PassionProfile,
    IntersectionIdea,
    SolutionIdea,
    ProfileBasedIdea,
)
from prompts import get_prompt

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


def call_llm(prompt: str, config: dict, system_prompt: str = None, schema=None) -> str:
    """
    Call configured LLM provider.

    This is a wrapper around the unified call_llm_provider that maintains
    the original function signature for backwards compatibility.

    Args:
        prompt: The user prompt
        config: Configuration dict
        system_prompt: Optional system prompt
        schema: Optional Pydantic schema for Ollama structured output

    Returns:
        Response text from the model
    """
    provider = config.get('api_provider', 'openai')

    # Build messages based on provider
    if provider == 'openai':
        # OpenAI: system prompt passed separately
        messages = [{"role": "user", "content": prompt}]
        pass_system_prompt = system_prompt
    elif provider == 'ollama':
        # Ollama: system prompt included in user content
        if system_prompt:
            full_content = f"{system_prompt}\n\n{prompt}"
        else:
            full_content = prompt
        messages = [{"role": "user", "content": full_content}]
        pass_system_prompt = None
    else:
        # Anthropic: system prompt included in user content
        if system_prompt:
            full_content = f"{system_prompt}\n\n{prompt}"
        else:
            full_content = prompt
        messages = [{"role": "user", "content": full_content}]
        pass_system_prompt = None

    return call_llm_provider(
        messages,
        config,
        schema=schema,
        max_tokens=8192,
        system_prompt=pass_system_prompt
    )


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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "passion_profile")

    total = profile_data.get("total_counts", {})
    prompt = prompt_template.format(
        total_ideas=total.get('ideas', 0),
        total_problems=total.get('problems', 0),
        total_workflows=total.get('workflows', 0),
        profile_data=json.dumps(profile_data, indent=2)
    )

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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "intersection_ideas")

    prompt = prompt_template.format(
        profile_summary=profile.get('summary', 'A developer exploring various interests.'),
        themes=json.dumps(themes, indent=2),
        tools=json.dumps(tools, indent=2)
    )

    # Use schema for Ollama structured output
    schema = make_array_schema(IntersectionIdea) if config.get('api_provider') == 'ollama' else None
    response = call_llm(prompt, config, system_prompt, schema=schema)
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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "solution_ideas")

    prompt = prompt_template.format(
        problems=json.dumps(problems, indent=2),
        tools=json.dumps(tools, indent=2),
        profile_summary=profile.get('summary', 'A developer exploring various interests.')
    )

    # Use schema for Ollama structured output
    schema = make_array_schema(SolutionIdea) if config.get('api_provider') == 'ollama' else None
    response = call_llm(prompt, config, system_prompt, schema=schema)
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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "profile_ideas")

    prompt = prompt_template.format(
        profile=json.dumps(profile, indent=2)
    )

    # Use schema for Ollama structured output
    schema = make_array_schema(ProfileBasedIdea) if config.get('api_provider') == 'ollama' else None
    response = call_llm(prompt, config, system_prompt, schema=schema)
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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "time_capsule")

    for item in old_high_passion:
        idea = item["idea"]
        months_ago = item["months_ago"]

        prompt = prompt_template.format(
            months_ago=months_ago,
            idea_name=idea.get('name', 'Unknown'),
            idea_description=idea.get('description', ''),
            motivations=json.dumps(idea.get('motivations', [])),
            passion_score=idea.get('scores', {}).get('passion', 0),
            first_date=item['first_date'],
            tools=json.dumps(all_tools[:20], indent=2)
        )

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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "deduplication")

    prompt = prompt_template.format(
        ideas=json.dumps(ideas_for_dedup, indent=2)
    )

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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "generated_scoring")

    prompt = prompt_template.format(
        ideas=json.dumps(ideas_for_scoring, indent=2),
        profile_summary=profile.get('summary', 'A developer with various interests.')
    )

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

    # Get prompt from config (with fallback to defaults)
    prompt_template, system_prompt = get_prompt(config, "project_development")

    prompt = prompt_template.format(
        idea=json.dumps(idea, indent=2),
        profile_summary=profile_summary
    )

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
