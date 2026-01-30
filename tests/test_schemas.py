"""
Tests for schemas.py - Pydantic data models.
"""
import pytest
from pydantic import ValidationError

from schemas import (
    ProjectIdea,
    Problem,
    Workflow,
    EmotionalSignals,
    ExtractionResult,
    EmptyExtractionResult,
    ConsolidatedIdea,
    ScoredIdea,
    CoreTheme,
    PassionProfile
)


def test_project_idea_valid():
    """Test creating a valid ProjectIdea."""
    idea = ProjectIdea(
        idea="Build a task manager",
        motivation="Need better organization",
        detail_level="sketched"
    )
    
    assert idea.idea == "Build a task manager"
    assert idea.motivation == "Need better organization"
    assert idea.detail_level == "sketched"


def test_project_idea_invalid_detail_level():
    """Test that invalid detail_level raises validation error."""
    with pytest.raises(ValidationError):
        ProjectIdea(
            idea="Test",
            motivation="Testing",
            detail_level="invalid"  # Should be vague/sketched/detailed
        )


def test_problem_valid():
    """Test creating a valid Problem."""
    problem = Problem(
        problem="Too many emails",
        context="Overwhelmed by inbox"
    )
    
    assert problem.problem == "Too many emails"
    assert problem.context == "Overwhelmed by inbox"


def test_workflow_valid():
    """Test creating a valid Workflow."""
    workflow = Workflow(
        workflow="Automated email sorting",
        status="building"
    )
    
    assert workflow.workflow == "Automated email sorting"
    assert workflow.status == "building"


def test_workflow_invalid_status():
    """Test that invalid status raises validation error."""
    with pytest.raises(ValidationError):
        Workflow(
            workflow="Test",
            status="invalid"  # Should be exploring/building/optimizing
        )


def test_emotional_signals_valid():
    """Test creating valid EmotionalSignals."""
    signals = EmotionalSignals(
        tone="excited",
        notes="Very motivated about this project"
    )
    
    assert signals.tone == "excited"
    assert signals.notes == "Very motivated about this project"


def test_emotional_signals_invalid_tone():
    """Test that invalid tone raises validation error."""
    with pytest.raises(ValidationError):
        EmotionalSignals(
            tone="angry",  # Should be excited/frustrated/curious/stuck/neutral
            notes="Test"
        )


def test_extraction_result_valid():
    """Test creating a valid ExtractionResult."""
    result = ExtractionResult(
        project_ideas=[
            ProjectIdea(idea="Test", motivation="Testing", detail_level="vague")
        ],
        problems=[],
        workflows=[],
        tools_explored=["Python", "Flask"],
        underlying_questions=["How to scale?"],
        emotional_signals=EmotionalSignals(tone="curious", notes="Exploring options")
    )
    
    assert len(result.project_ideas) == 1
    assert len(result.tools_explored) == 2
    assert result.emotional_signals.tone == "curious"


def test_extraction_result_defaults_to_empty_lists():
    """Test that ExtractionResult defaults empty lists."""
    result = ExtractionResult(
        emotional_signals=EmotionalSignals(tone="neutral", notes="")
    )
    
    assert result.project_ideas == []
    assert result.problems == []
    assert result.workflows == []
    assert result.tools_explored == []
    assert result.underlying_questions == []


def test_empty_extraction_result():
    """Test EmptyExtractionResult for conversations with nothing extractable."""
    result = EmptyExtractionResult(
        empty=True,
        reason="Just casual chat, no projects or problems mentioned"
    )
    
    assert result.empty is True
    assert "casual chat" in result.reason


def test_consolidated_idea_valid():
    """Test creating a valid ConsolidatedIdea."""
    idea = ConsolidatedIdea(
        name="Task Manager App",
        description="A productivity app with tags and priorities",
        occurrences=3,
        date_range=["2023-12-01", "2023-12-22"],
        evolution="Started vague, became more detailed over time",
        source_ids=["conv-1", "conv-2", "conv-3"],
        motivations=["Need organization", "Current tools are bad"],
        detail_levels=["vague", "sketched", "detailed"]
    )
    
    assert idea.name == "Task Manager App"
    assert idea.occurrences == 3
    assert len(idea.source_ids) == 3


def test_scored_idea_valid():
    """Test creating a valid ScoredIdea."""
    idea = ScoredIdea(
        name="Quick prototype",
        effort=2,
        monetization=1,
        personal_utility=5,
        reasoning="Weekend project, no revenue, but very useful to me"
    )
    
    assert idea.effort == 2
    assert idea.monetization == 1
    assert idea.personal_utility == 5


def test_scored_idea_invalid_scores():
    """Test that scores outside 1-5 range raise validation error."""
    with pytest.raises(ValidationError):
        ScoredIdea(
            name="Test",
            effort=0,  # Should be 1-5
            monetization=1,
            personal_utility=1,
            reasoning=""
        )
    
    with pytest.raises(ValidationError):
        ScoredIdea(
            name="Test",
            effort=1,
            monetization=6,  # Should be 1-5
            personal_utility=1,
            reasoning=""
        )


def test_core_theme_valid():
    """Test creating a valid CoreTheme."""
    theme = CoreTheme(
        theme="Productivity tools",
        strength=4,
        evidence="Mentioned in 15 conversations"
    )
    
    assert theme.theme == "Productivity tools"
    assert theme.strength == 4


def test_core_theme_invalid_strength():
    """Test that strength outside 1-5 raises validation error."""
    with pytest.raises(ValidationError):
        CoreTheme(
            theme="Test",
            strength=0,
            evidence=""
        )


def test_passion_profile_valid():
    """Test creating a valid PassionProfile."""
    profile = PassionProfile(
        core_themes=[
            CoreTheme(theme="AI", strength=5, evidence="20 convos")
        ],
        emotional_patterns={
            "excited_about": ["ML", "automation"],
            "frustrated_by": ["manual work"],
            "curious_about": ["LLMs"]
        },
        summary="User passionate about AI and automation"
    )
    
    assert len(profile.core_themes) == 1
    assert "AI" in profile.summary


def test_passion_profile_defaults():
    """Test PassionProfile with default empty lists."""
    profile = PassionProfile(
        emotional_patterns={
            "excited_about": [],
            "frustrated_by": [],
            "curious_about": []
        },
        summary="Minimal profile"
    )
    
    assert profile.core_themes == []
    assert profile.tool_expertise == []
    assert profile.high_passion_ideas == []


def test_extraction_result_to_dict():
    """Test that ExtractionResult can be serialized to dict."""
    result = ExtractionResult(
        project_ideas=[
            ProjectIdea(idea="Test", motivation="Testing", detail_level="vague")
        ],
        emotional_signals=EmotionalSignals(tone="neutral", notes="")
    )
    
    data = result.model_dump()
    
    assert isinstance(data, dict)
    assert "project_ideas" in data
    assert len(data["project_ideas"]) == 1


def test_extraction_result_from_json():
    """Test creating ExtractionResult from JSON dict."""
    data = {
        "project_ideas": [
            {"idea": "Test", "motivation": "Testing", "detail_level": "vague"}
        ],
        "problems": [],
        "workflows": [],
        "tools_explored": [],
        "underlying_questions": [],
        "emotional_signals": {
            "tone": "neutral",
            "notes": ""
        }
    }
    
    result = ExtractionResult(**data)
    
    assert len(result.project_ideas) == 1
    assert result.project_ideas[0].idea == "Test"
