#!/usr/bin/env python3
"""
End-to-end test of Ollama integration with real Resurface data.
Tests the full pipeline: Extraction -> Consolidation -> Categorization -> Synthesis
"""
import json
import random
from pathlib import Path

from ollama_client import chat_with_system, make_array_schema, check_connection
from prompts import (
    build_extraction_prompt,
    build_consolidation_prompt,
    build_scoring_prompt,
    build_intersection_prompt,
    build_solution_prompt,
    SYSTEM_PROMPTS,
)
from schemas import (
    ExtractionResult,
    ConsolidatedIdea,
    ConsolidatedProblem,
    ScoredIdea,
    IntersectionIdea,
    SolutionIdea,
)

# Paths
RESURFACE_DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL = "gemma3:4b"
NUM_CONVERSATIONS = 7


def load_manifest():
    """Load conversation manifest."""
    path = RESURFACE_DATA_DIR / "parsed" / "_manifest.json"
    with open(path) as f:
        return json.load(f).get("conversations", [])


def load_conversation(conv_id: str) -> dict:
    """Load a specific conversation."""
    path = RESURFACE_DATA_DIR / "parsed" / f"{conv_id}.json"
    with open(path) as f:
        return json.load(f)


def format_conversation(conv: dict) -> str:
    """Format conversation for prompt."""
    lines = [f"Title: {conv.get('title', 'Untitled')}"]
    lines.append(f"Date: {conv.get('created', 'Unknown')[:10]}")
    lines.append("")

    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:2000]  # Truncate long messages
        lines.append(f"{role}: {content}")
        lines.append("")

    return "\n".join(lines)


def run_extraction(conv: dict) -> dict:
    """Run extraction on a single conversation."""
    conv_text = format_conversation(conv)
    prompt = build_extraction_prompt(conv_text)

    print(f"\n{'='*60}")
    print(f"EXTRACTION: {conv.get('title', 'Untitled')[:50]}")
    print(f"Messages: {len(conv.get('messages', []))}")
    print(f"{'='*60}")

    response, duration = chat_with_system(
        model=MODEL,
        system_prompt=SYSTEM_PROMPTS["extraction"],
        user_prompt=prompt,
        schema=ExtractionResult,
        temperature=0.0
    )

    print(f"Time: {duration:.1f}s")

    result = json.loads(response)

    # Show key extractions
    ideas = result.get("project_ideas", [])
    problems = result.get("problems", [])
    tools = result.get("tools_explored", [])
    emotional = result.get("emotional_signals", {})

    print(f"\nExtracted:")
    print(f"  Ideas: {len(ideas)}")
    for idea in ideas[:3]:
        print(f"    - {idea.get('idea', '')[:60]}")

    print(f"  Problems: {len(problems)}")
    for prob in problems[:2]:
        print(f"    - {prob.get('problem', '')[:60]}")

    print(f"  Tools: {tools[:5]}")
    print(f"  Tone: {emotional.get('tone', 'unknown')}")

    return result


def run_consolidation(all_extractions: list) -> dict:
    """Consolidate all extracted ideas and problems."""
    print(f"\n{'='*60}")
    print("CONSOLIDATION")
    print(f"{'='*60}")

    # Collect all ideas with source info
    all_ideas = []
    all_problems = []

    for i, ext in enumerate(all_extractions):
        for idea in ext.get("project_ideas", []):
            idea["_source_id"] = f"conv-{i}"
            all_ideas.append(idea)
        for prob in ext.get("problems", []):
            prob["_source_id"] = f"conv-{i}"
            all_problems.append(prob)

    print(f"Total ideas to consolidate: {len(all_ideas)}")
    print(f"Total problems to consolidate: {len(all_problems)}")

    results = {}

    # Consolidate ideas
    if all_ideas:
        prompt = build_consolidation_prompt("ideas", all_ideas)
        response, duration = chat_with_system(
            model=MODEL,
            system_prompt=SYSTEM_PROMPTS["consolidation"],
            user_prompt=prompt,
            schema=make_array_schema(ConsolidatedIdea),
            temperature=0.0
        )
        print(f"\nIdeas consolidation: {duration:.1f}s")

        consolidated_ideas = json.loads(response)
        results["ideas"] = consolidated_ideas

        print(f"Consolidated into {len(consolidated_ideas)} clusters:")
        for idea in consolidated_ideas[:3]:
            print(f"  - {idea.get('name', '')}: {idea.get('description', '')[:50]}...")
            print(f"    Occurrences: {idea.get('occurrences', 0)}")

    # Consolidate problems
    if all_problems:
        prompt = build_consolidation_prompt("problems", all_problems)
        response, duration = chat_with_system(
            model=MODEL,
            system_prompt=SYSTEM_PROMPTS["consolidation"],
            user_prompt=prompt,
            schema=make_array_schema(ConsolidatedProblem),
            temperature=0.0
        )
        print(f"\nProblems consolidation: {duration:.1f}s")

        consolidated_problems = json.loads(response)
        results["problems"] = consolidated_problems

        print(f"Consolidated into {len(consolidated_problems)} clusters:")
        for prob in consolidated_problems[:3]:
            print(f"  - {prob.get('name', '')}: {prob.get('description', '')[:50]}...")

    return results


def run_categorization(consolidated_ideas: list) -> list:
    """Score and categorize consolidated ideas."""
    print(f"\n{'='*60}")
    print("CATEGORIZATION")
    print(f"{'='*60}")

    if not consolidated_ideas:
        print("No ideas to categorize")
        return []

    prompt = build_scoring_prompt(consolidated_ideas)
    response, duration = chat_with_system(
        model=MODEL,
        system_prompt=SYSTEM_PROMPTS["categorization"],
        user_prompt=prompt,
        schema=make_array_schema(ScoredIdea),
        temperature=0.0
    )

    print(f"Time: {duration:.1f}s")

    scored = json.loads(response)

    print(f"\nScored {len(scored)} ideas:")
    for idea in scored:
        print(f"  - {idea.get('name', '')}")
        print(f"    Effort: {idea.get('effort')}/5, Monetization: {idea.get('monetization')}/5, Utility: {idea.get('personal_utility')}/5")
        print(f"    Reasoning: {idea.get('reasoning', '')[:80]}...")

    return scored


def run_synthesis(consolidated: dict, scored_ideas: list) -> dict:
    """Generate new ideas via synthesis."""
    print(f"\n{'='*60}")
    print("SYNTHESIS")
    print(f"{'='*60}")

    # Build a simple profile summary from the data
    idea_names = [i.get("name", "") for i in consolidated.get("ideas", [])]
    problem_names = [p.get("name", "") for p in consolidated.get("problems", [])]

    profile_summary = f"A developer interested in: {', '.join(idea_names[:3])}. Struggles with: {', '.join(problem_names[:2])}."

    themes = idea_names[:5]
    tools = ["Python", "JavaScript", "CLI tools", "automation"]  # Example tools
    problems = consolidated.get("problems", [])[:5]

    results = {}

    # Intersection ideas
    print("\nGenerating intersection ideas...")
    prompt = build_intersection_prompt(profile_summary, themes, tools)
    response, duration = chat_with_system(
        model=MODEL,
        system_prompt=SYSTEM_PROMPTS["synthesis_intersection"],
        user_prompt=prompt,
        schema=make_array_schema(IntersectionIdea),
        temperature=0.0
    )

    print(f"Time: {duration:.1f}s")
    intersection_ideas = json.loads(response)
    results["intersection"] = intersection_ideas

    print(f"Generated {len(intersection_ideas)} intersection ideas:")
    for idea in intersection_ideas[:3]:
        print(f"  - {idea.get('name', '')}")
        print(f"    {idea.get('description', '')[:80]}...")
        print(f"    Why exciting: {idea.get('why_exciting', '')[:60]}...")

    # Solution ideas
    print("\nGenerating solution ideas...")
    prompt = build_solution_prompt(profile_summary, problems, tools)
    response, duration = chat_with_system(
        model=MODEL,
        system_prompt=SYSTEM_PROMPTS["synthesis_solution"],
        user_prompt=prompt,
        schema=make_array_schema(SolutionIdea),
        temperature=0.0
    )

    print(f"Time: {duration:.1f}s")
    solution_ideas = json.loads(response)
    results["solution"] = solution_ideas

    print(f"Generated {len(solution_ideas)} solution ideas:")
    for idea in solution_ideas[:3]:
        print(f"  - {idea.get('name', '')}")
        print(f"    {idea.get('description', '')[:80]}...")
        print(f"    Problem addressed: {idea.get('problem_addressed', '')[:60]}...")

    return results


def main():
    print("="*60)
    print("END-TO-END OLLAMA TEST WITH REAL DATA")
    print(f"Model: {MODEL}")
    print(f"Conversations: {NUM_CONVERSATIONS}")
    print("="*60)

    # Check connection
    if not check_connection():
        print("ERROR: Cannot connect to Ollama. Is it running?")
        return

    print("Connected to Ollama")

    # Load manifest and select random conversations
    manifest = load_manifest()
    print(f"Total conversations available: {len(manifest)}")

    # Select conversations with reasonable message counts (10-100 messages)
    suitable = [c for c in manifest if 10 <= c.get("message_count", 0) <= 100]
    print(f"Suitable conversations (10-100 msgs): {len(suitable)}")

    selected = random.sample(suitable, min(NUM_CONVERSATIONS, len(suitable)))

    print(f"\nSelected {len(selected)} conversations:")
    for c in selected:
        print(f"  - {c.get('title', 'Untitled')[:50]} ({c.get('message_count', 0)} msgs)")

    # Phase 1: Extraction
    print("\n" + "="*60)
    print("PHASE 1: EXTRACTION")
    print("="*60)

    all_extractions = []
    for conv_meta in selected:
        conv = load_conversation(conv_meta["id"])
        try:
            extraction = run_extraction(conv)
            extraction["_source_title"] = conv.get("title", "")
            all_extractions.append(extraction)
        except Exception as e:
            print(f"Error extracting: {e}")

    print(f"\nSuccessfully extracted from {len(all_extractions)} conversations")

    # Phase 2: Consolidation
    print("\n" + "="*60)
    print("PHASE 2: CONSOLIDATION")
    print("="*60)

    consolidated = run_consolidation(all_extractions)

    # Phase 3: Categorization
    print("\n" + "="*60)
    print("PHASE 3: CATEGORIZATION")
    print("="*60)

    scored_ideas = run_categorization(consolidated.get("ideas", []))

    # Phase 4: Synthesis
    print("\n" + "="*60)
    print("PHASE 4: SYNTHESIS")
    print("="*60)

    synthesis = run_synthesis(consolidated, scored_ideas)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Conversations processed: {len(all_extractions)}")
    print(f"Ideas extracted: {sum(len(e.get('project_ideas', [])) for e in all_extractions)}")
    print(f"Problems extracted: {sum(len(e.get('problems', [])) for e in all_extractions)}")
    print(f"Consolidated idea clusters: {len(consolidated.get('ideas', []))}")
    print(f"Consolidated problem clusters: {len(consolidated.get('problems', []))}")
    print(f"Scored ideas: {len(scored_ideas)}")
    print(f"Intersection ideas generated: {len(synthesis.get('intersection', []))}")
    print(f"Solution ideas generated: {len(synthesis.get('solution', []))}")
    print("\nTest complete!")


if __name__ == "__main__":
    main()
