"""
Pydantic schemas matching Resurface's expected LLM output formats.

These schemas are used both for:
1. Passing to Ollama's `format` parameter for structured output enforcement
2. Validating responses match expected structure
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# EXTRACTION SCHEMAS (from extractor.py)
# =============================================================================

class ProjectIdea(BaseModel):
    """A project idea extracted from a conversation."""
    idea: str = Field(description="Description of the project idea")
    motivation: str = Field(description="Why the user wants to build this")
    detail_level: Literal["vague", "sketched", "detailed"] = Field(
        description="How detailed the idea is"
    )


class Problem(BaseModel):
    """A problem or frustration mentioned in conversation."""
    problem: str = Field(description="The problem or pain point")
    context: str = Field(description="Context around the problem")


class Workflow(BaseModel):
    """A workflow or automation being explored."""
    workflow: str = Field(description="Description of the workflow")
    status: Literal["exploring", "building", "optimizing"] = Field(
        description="Current status of the workflow"
    )


class EmotionalSignals(BaseModel):
    """Emotional tone detected in conversation."""
    tone: Literal["excited", "frustrated", "curious", "stuck", "neutral"] = Field(
        description="Overall emotional tone"
    )
    notes: str = Field(description="Brief notes about emotional signals")


class ExtractionResult(BaseModel):
    """Complete extraction result from a conversation."""
    project_ideas: list[ProjectIdea] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    workflows: list[Workflow] = Field(default_factory=list)
    tools_explored: list[str] = Field(default_factory=list)
    underlying_questions: list[str] = Field(default_factory=list)
    emotional_signals: EmotionalSignals


class EmptyExtractionResult(BaseModel):
    """Result when conversation has nothing extractable."""
    empty: Literal[True] = True
    reason: str = Field(description="Why nothing was extracted")


# =============================================================================
# CONSOLIDATION SCHEMAS (from consolidate.py)
# =============================================================================

class ConsolidatedIdea(BaseModel):
    """A consolidated project idea from multiple conversations."""
    name: str = Field(description="Clear consolidated name")
    description: str = Field(description="2-3 sentence synthesis")
    occurrences: int = Field(description="How many times it appeared")
    date_range: list[str] = Field(description="[earliest, latest] dates")
    evolution: str = Field(description="How the idea evolved over time")
    source_ids: list[str] = Field(description="Conversation IDs containing this idea")
    motivations: list[str] = Field(description="Combined motivations")
    detail_levels: list[str] = Field(description="Detail levels from each mention")


class ConsolidatedProblem(BaseModel):
    """A consolidated problem from multiple conversations."""
    name: str = Field(description="Clear consolidated name")
    description: str = Field(description="2-3 sentence synthesis")
    occurrences: int = Field(description="How many times it appeared")
    date_range: list[str] = Field(description="[earliest, latest] dates")
    source_ids: list[str] = Field(description="Conversation IDs")
    contexts: list[str] = Field(description="Combined contexts")


class ConsolidatedWorkflow(BaseModel):
    """A consolidated workflow from multiple conversations."""
    name: str = Field(description="Clear consolidated name")
    description: str = Field(description="2-3 sentence synthesis")
    occurrences: int = Field(description="How many times it appeared")
    date_range: list[str] = Field(description="[earliest, latest] dates")
    source_ids: list[str] = Field(description="Conversation IDs")
    statuses: list[str] = Field(description="Statuses from each mention")


# =============================================================================
# CATEGORIZATION SCHEMAS (from categorize.py)
# =============================================================================

class ScoredIdea(BaseModel):
    """An idea with effort/monetization/utility scores."""
    name: str = Field(description="Exact name from input")
    effort: int = Field(ge=1, le=5, description="1=weekend, 5=year+")
    monetization: int = Field(ge=1, le=5, description="1=no revenue, 5=high potential")
    personal_utility: int = Field(ge=1, le=5, description="1=nice to have, 5=essential")
    reasoning: str = Field(description="Brief explanation of scores")


# =============================================================================
# SYNTHESIS SCHEMAS (from synthesizer.py)
# =============================================================================

class CoreTheme(BaseModel):
    """A core theme from user's interests."""
    theme: str = Field(description="The theme name")
    strength: int = Field(ge=1, le=5, description="How strong this theme is")
    evidence: str = Field(description="Evidence from the data")


class ToolExpertise(BaseModel):
    """A tool the user has expertise with."""
    tool: str = Field(description="Tool name")
    mentions: int = Field(description="Number of mentions")
    contexts: list[str] = Field(description="Contexts where mentioned")


class RecurringProblem(BaseModel):
    """A recurring problem."""
    problem: str = Field(description="Problem description")
    frequency: int = Field(description="How often it appears")
    impact: str = Field(description="Impact description")


class EmotionalPatterns(BaseModel):
    """Emotional patterns from conversations."""
    excited_about: list[str] = Field(default_factory=list)
    frustrated_by: list[str] = Field(default_factory=list)
    curious_about: list[str] = Field(default_factory=list)


class HighPassionIdea(BaseModel):
    """An idea with high passion score."""
    name: str = Field(description="Idea name")
    passion_score: int = Field(ge=1, le=5, description="Passion level")


class PassionProfile(BaseModel):
    """Complete passion profile synthesized from user data."""
    core_themes: list[CoreTheme] = Field(default_factory=list)
    tool_expertise: list[ToolExpertise] = Field(default_factory=list)
    recurring_problems: list[RecurringProblem] = Field(default_factory=list)
    emotional_patterns: EmotionalPatterns
    underlying_questions: list[str] = Field(default_factory=list)
    high_passion_ideas: list[HighPassionIdea] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence description")


class IntersectionIdea(BaseModel):
    """An idea generated at intersection of themes."""
    name: str = Field(description="Catchy project name")
    description: str = Field(description="2-3 sentences")
    themes_combined: list[str] = Field(description="Which themes this intersects")
    tools_suggested: list[str] = Field(description="Relevant tools")
    why_exciting: str = Field(description="Why this person would love it")


class SolutionIdea(BaseModel):
    """An idea that solves a recurring problem."""
    name: str = Field(description="Clear project name")
    description: str = Field(description="What it does and how")
    problem_addressed: str = Field(description="Which problem it solves")
    tools_used: list[str] = Field(description="Tools from user's toolkit")
    why_practical: str = Field(description="Why it's achievable")


class ProfileBasedIdea(BaseModel):
    """An idea generated from holistic profile understanding."""
    name: str = Field(description="Project name")
    description: str = Field(description="What it does")
    profile_alignment: str = Field(description="How it aligns with profile")
    novelty_factor: str = Field(description="What makes it novel")
