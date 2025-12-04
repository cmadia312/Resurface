#!/usr/bin/env python3
"""
Default prompts registry for Resurface.

All LLM prompts are defined here with metadata.
Users can customize prompts via config.json.
"""

DEFAULT_PROMPTS = {
    # === EXTRACTION PHASE ===
    "extraction": {
        "name": "Extraction",
        "description": "Extracts insights from conversations (ideas, problems, workflows, tools, emotions)",
        "template": """Analyze this conversation and extract the following. Be specific—include enough context that someone reading this extraction would understand the idea without the original. If a category has nothing notable, omit it.

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
{"empty": true, "reason": "..."}""",
        "system_prompt": "",
        "variables": ["conversation"]
    },

    "extraction_retry": {
        "name": "Extraction Retry",
        "description": "Fallback prompt when JSON parsing fails",
        "template": "Your previous response was not valid JSON. Please return only valid JSON matching the schema.",
        "system_prompt": "",
        "variables": []
    },

    # === CONSOLIDATION PHASE ===
    "consolidate_ideas": {
        "name": "Consolidate Ideas",
        "description": "Groups duplicate/similar project ideas across conversations",
        "template": """Here are project ideas extracted from multiple conversations over time.
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
        "system_prompt": "You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions.",
        "variables": ["items"]
    },

    "consolidate_problems": {
        "name": "Consolidate Problems",
        "description": "Groups duplicate/similar problems across conversations",
        "template": """Here are problems/frustrations extracted from multiple conversations over time.
Group them into unique themes—merge duplicates and near-duplicates that represent the same underlying issue.

For each unique theme, return:
- name: A clear consolidated name
- description: 2-3 sentence synthesis of all mentions
- occurrences: How many times it appeared
- date_range: [earliest_mention, latest_mention]
- source_ids: List of conversation_ids that contained this problem
- contexts: Combined list of contexts from all mentions

Return as a JSON array. Only return valid JSON, no other text.""",
        "system_prompt": "You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions.",
        "variables": ["items"]
    },

    "consolidate_workflows": {
        "name": "Consolidate Workflows",
        "description": "Groups duplicate/similar workflows across conversations",
        "template": """Here are workflows/automations extracted from multiple conversations over time.
Group them into unique concepts—merge duplicates and near-duplicates that represent the same workflow.

For each unique concept, return:
- name: A clear consolidated name
- description: 2-3 sentence synthesis of all mentions
- occurrences: How many times it appeared
- date_range: [earliest_mention, latest_mention]
- source_ids: List of conversation_ids that contained this workflow
- statuses: List of statuses from each mention (exploring/building/optimizing)

Return as a JSON array. Only return valid JSON, no other text.""",
        "system_prompt": "You are a data consolidation assistant. Return only valid JSON arrays. Be concise in descriptions.",
        "variables": ["items"]
    },

    # === CATEGORIZATION PHASE ===
    "scoring": {
        "name": "Idea Scoring",
        "description": "Scores consolidated ideas on effort, monetization, and utility",
        "template": """Score these project ideas on three dimensions. Use 1-5 scale.

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

Return only valid JSON.""",
        "system_prompt": "You are a startup advisor scoring project ideas. Return only valid JSON.",
        "variables": ["ideas_json"]
    },

    # === SYNTHESIS PHASE ===
    "passion_profile": {
        "name": "Passion Profile",
        "description": "Synthesizes user interests into a comprehensive passion profile",
        "template": """Analyze this user's conversation history data and build a passion profile.

DATA PROVIDED (sampled from {total_ideas} ideas, {total_problems} problems, {total_workflows} workflows):
- Top Ideas (highest scoring project ideas with categories and passion/recurrence scores)
- Top Problems (pain points)
- Top Workflows (automations and processes)
- Tool Frequency (top technologies mentioned)
- Emotional Samples (conversation tones over time)

{profile_data}

Synthesize this into a passion profile with:
1. core_themes: Top 5-10 recurring themes/interests (with evidence from the data)
2. tool_expertise: Tools they use/explore most (with contexts)
3. recurring_problems: Pain points that keep appearing
4. emotional_patterns: What excites them vs frustrates them vs makes them curious
5. underlying_questions: Deeper uncertainties they keep exploring
6. high_passion_ideas: Ideas with highest passion scores
7. summary: 2-3 sentence description of this person's interests/focus

Return JSON matching this schema:
{{
  "core_themes": [{{"theme": "string", "strength": 1-5, "evidence": "string"}}],
  "tool_expertise": [{{"tool": "string", "mentions": number, "contexts": ["string"]}}],
  "recurring_problems": [{{"problem": "string", "frequency": number, "impact": "string"}}],
  "emotional_patterns": {{
    "excited_about": ["string"],
    "frustrated_by": ["string"],
    "curious_about": ["string"]
  }},
  "underlying_questions": ["string"],
  "high_passion_ideas": [{{"name": "string", "passion_score": 1-5}}],
  "summary": "2-3 sentence description"
}}""",
        "system_prompt": "You are an expert at analyzing patterns in human interests and synthesizing insights. Return only valid JSON.",
        "variables": ["total_ideas", "total_problems", "total_workflows", "profile_data"]
    },

    "intersection_ideas": {
        "name": "Intersection Ideas",
        "description": "Generates ideas at the intersection of user's themes",
        "template": """You are helping someone discover project ideas at the intersection of their passions.

USER PROFILE SUMMARY:
{profile_summary}

THEIR TOP THEMES:
{themes}

TOOLS THEY KNOW:
{tools}

Generate 5 NOVEL project ideas that creatively combine 2-3 of these themes.
These should be ideas the user has NOT explicitly mentioned - synthesize something new.

For each idea provide:
- name: Catchy, memorable project name
- description: 2-3 sentences explaining what it does
- themes_combined: Which themes this intersects
- tools_suggested: Relevant tools from their expertise
- why_exciting: Why this person specifically would love building this

Return JSON array:
[{{"name": "...", "description": "...", "themes_combined": [...], "tools_suggested": [...], "why_exciting": "..."}}]""",
        "system_prompt": "You are a creative project ideation expert. Generate novel, exciting project ideas. Return only valid JSON array.",
        "variables": ["profile_summary", "themes", "tools"]
    },

    "solution_ideas": {
        "name": "Solution Ideas",
        "description": "Generates solutions to user's recurring problems using their tools",
        "template": """You are helping someone solve their recurring problems using tools they already know.

USER'S RECURRING PROBLEMS:
{problems}

TOOLS THEY'RE FAMILIAR WITH:
{tools}

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
[{{"name": "...", "description": "...", "problem_addressed": "...", "tools_used": [...], "why_practical": "..."}}]""",
        "system_prompt": "You are a practical problem-solving expert. Generate actionable, buildable solutions. Return only valid JSON array.",
        "variables": ["problems", "tools", "profile_summary"]
    },

    "profile_ideas": {
        "name": "Profile-Based Ideas",
        "description": "Generates ideas based on holistic user profile understanding",
        "template": """You are roleplaying as someone with the following passion profile:

{profile}

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
[{{"name": "...", "description": "...", "profile_alignment": "...", "novelty_factor": "..."}}]""",
        "system_prompt": "You are an empathetic ideation expert who deeply understands human interests. Generate surprising, delightful ideas. Return only valid JSON array.",
        "variables": ["profile"]
    },

    "time_capsule": {
        "name": "Time Capsule",
        "description": "Resurfaces old high-passion ideas with updated context",
        "template": """This user had an idea {months_ago} months ago that they were excited about:

ORIGINAL IDEA:
Name: {idea_name}
Description: {idea_description}
Motivation: {motivations}
Passion Score: {passion_score}/5
Date First Mentioned: {first_date}

TOOLS THEY'VE EXPLORED:
{tools}

Write two things:

1. A brief "letter from past self" (2-3 sentences) - what past-them would say about why this mattered

2. An UPDATED VERSION of this idea that could incorporate their knowledge/tools

Return JSON:
{{
  "letter_from_past": "...",
  "updated_name": "...",
  "updated_vision": "How to approach this now with new knowledge",
  "tools_to_use": ["..."]
}}""",
        "system_prompt": "You are helping someone reconnect with their past ideas. Be warm and encouraging. Return only valid JSON.",
        "variables": ["months_ago", "idea_name", "idea_description", "motivations", "passion_score", "first_date", "tools"]
    },

    "deduplication": {
        "name": "Deduplication",
        "description": "Identifies and merges semantically similar generated ideas",
        "template": """Review these generated project ideas and identify any that are essentially the same idea.

IDEAS:
{ideas}

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
{{"keep": [], "unique": ["all", "the", "ids"]}}""",
        "system_prompt": "You are an expert at identifying semantic similarity. Return only valid JSON.",
        "variables": ["ideas"]
    },

    "generated_scoring": {
        "name": "Generated Ideas Scoring",
        "description": "Scores generated ideas on 5 dimensions",
        "template": """Score these generated project ideas on 5 dimensions (1-5 scale each):

IDEAS:
{ideas}

USER PROFILE SUMMARY:
{profile_summary}

For each idea, score:
1. effort: 1=weekend, 2=week, 3=month, 4=few months, 5=year+
2. monetization: 1=none, 2=small niche, 3=moderate, 4=solid business, 5=high potential
3. personal_utility: 1=nice to have, 3=regularly useful, 5=essential
4. passion_alignment: 1=low match, 3=moderate, 5=perfect match to their interests
5. novelty: 1=common idea, 3=somewhat fresh, 5=highly original

Return JSON array:
[{{"id": "...", "effort": N, "monetization": N, "personal_utility": N, "passion_alignment": N, "novelty": N}}]""",
        "system_prompt": "You are an expert at evaluating project ideas. Return only valid JSON array.",
        "variables": ["ideas", "profile_summary"]
    },

    "project_development": {
        "name": "Project Development",
        "description": "Generates detailed project specification for an idea",
        "template": """Develop a detailed project specification for this idea:

IDEA:
{idea}

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
}}""",
        "system_prompt": "You are an expert project planner. Create actionable, realistic project specifications. Return only valid JSON.",
        "variables": ["idea", "profile_summary"]
    },
}


def get_prompt(config: dict, prompt_key: str) -> tuple[str, str]:
    """
    Get prompt template and system prompt, checking config for customizations.

    Args:
        config: Configuration dict (may contain 'prompts' section)
        prompt_key: Key identifying which prompt to retrieve

    Returns:
        Tuple of (template, system_prompt)

    Raises:
        KeyError: If prompt_key is not found in defaults
    """
    if prompt_key not in DEFAULT_PROMPTS:
        raise KeyError(f"Unknown prompt key: {prompt_key}")

    # Check for custom prompt in config
    custom = config.get("prompts", {}).get(prompt_key)
    if custom:
        return custom.get("template", ""), custom.get("system_prompt", "")

    # Return default
    default = DEFAULT_PROMPTS[prompt_key]
    return default["template"], default.get("system_prompt", "")


def get_prompt_metadata(prompt_key: str) -> dict:
    """
    Get metadata about a prompt (name, description, variables).

    Args:
        prompt_key: Key identifying which prompt

    Returns:
        Dict with name, description, variables
    """
    if prompt_key not in DEFAULT_PROMPTS:
        return {}

    prompt = DEFAULT_PROMPTS[prompt_key]
    return {
        "name": prompt.get("name", prompt_key),
        "description": prompt.get("description", ""),
        "variables": prompt.get("variables", [])
    }


def get_all_prompt_keys() -> list[str]:
    """Get list of all available prompt keys."""
    return list(DEFAULT_PROMPTS.keys())


def get_default_prompt(prompt_key: str) -> tuple[str, str]:
    """
    Get the default prompt (ignoring any customizations).

    Args:
        prompt_key: Key identifying which prompt

    Returns:
        Tuple of (template, system_prompt)
    """
    if prompt_key not in DEFAULT_PROMPTS:
        raise KeyError(f"Unknown prompt key: {prompt_key}")

    default = DEFAULT_PROMPTS[prompt_key]
    return default["template"], default.get("system_prompt", "")
