"""
Prompts copied from Resurface for testing Ollama models.

These are the exact prompts used in production, extracted from:
- extractor.py
- consolidate.py
- categorize.py
- synthesizer.py
"""
import json

# =============================================================================
# EXTRACTION PROMPT (from extractor.py:11-36)
# =============================================================================

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


# =============================================================================
# CONSOLIDATION PROMPTS (from consolidate.py:49-90)
# =============================================================================

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


# =============================================================================
# CATEGORIZATION PROMPT (from categorize.py:58-94)
# =============================================================================

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


# =============================================================================
# SYNTHESIS PROMPTS (from synthesizer.py)
# =============================================================================

INTERSECTION_PROMPT_TEMPLATE = """You are helping someone discover project ideas at the intersection of their passions.

USER PROFILE SUMMARY:
{profile_summary}

THEIR TOP THEMES:
{themes_json}

TOOLS THEY KNOW:
{tools_json}

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


SOLUTION_PROMPT_TEMPLATE = """You are helping someone solve their recurring problems using tools they already know.

USER'S RECURRING PROBLEMS:
{problems_json}

TOOLS THEY'RE FAMILIAR WITH:
{tools_json}

USER PROFILE SUMMARY:
{profile_summary}

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


# =============================================================================
# HELPER FUNCTIONS FOR BUILDING TEST PROMPTS
# =============================================================================

def build_extraction_prompt(conversation_text: str) -> str:
    """Build full extraction prompt with conversation."""
    return f"{EXTRACTION_PROMPT}\n\n---\n\n{conversation_text}"


def build_consolidation_prompt(item_type: str, items: list[dict]) -> str:
    """Build consolidation prompt with items to consolidate."""
    base_prompt = CONSOLIDATION_PROMPTS.get(item_type)
    if not base_prompt:
        raise ValueError(f"Unknown item type: {item_type}")
    items_text = json.dumps(items, indent=2)
    return f"{base_prompt}\n\nItems to consolidate:\n{items_text}"


def build_scoring_prompt(ideas: list[dict]) -> str:
    """Build scoring prompt with ideas to score."""
    ideas_for_prompt = []
    for idea in ideas:
        ideas_for_prompt.append({
            "name": idea.get("name", "Unknown"),
            "description": idea.get("description", ""),
            "motivations": idea.get("motivations", []),
            "occurrences": idea.get("occurrences", 1),
            "evolution": idea.get("evolution", "")
        })
    return SCORING_PROMPT.format(ideas_json=json.dumps(ideas_for_prompt, indent=2))


def build_intersection_prompt(
    profile_summary: str,
    themes: list[str],
    tools: list[str]
) -> str:
    """Build intersection ideas prompt."""
    return INTERSECTION_PROMPT_TEMPLATE.format(
        profile_summary=profile_summary,
        themes_json=json.dumps(themes, indent=2),
        tools_json=json.dumps(tools, indent=2)
    )


def build_solution_prompt(
    profile_summary: str,
    problems: list[dict],
    tools: list[str]
) -> str:
    """Build solution ideas prompt."""
    return SOLUTION_PROMPT_TEMPLATE.format(
        profile_summary=profile_summary,
        problems_json=json.dumps(problems, indent=2),
        tools_json=json.dumps(tools, indent=2)
    )


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS = {
    "extraction": "You are a conversation analyst extracting insights. Return only valid JSON.",
    "consolidation": "You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions.",
    "categorization": "You are a startup advisor scoring project ideas. Return only valid JSON.",
    "synthesis_intersection": "You are a creative project ideation expert. Generate novel, exciting project ideas. Return only valid JSON array.",
    "synthesis_solution": "You are a practical problem-solving expert. Generate actionable, buildable solutions. Return only valid JSON array.",
}
