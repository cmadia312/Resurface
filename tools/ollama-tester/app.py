#!/usr/bin/env python3
"""
Ollama LLM Testing Application for Resurface.

Tests local Ollama models against Resurface's structured output requirements.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from ollama_client import (
    check_connection,
    get_model_names,
    chat_with_system,
    make_array_schema,
    OllamaConnectionError,
    OllamaError,
)
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
from validator import (
    validate_extraction_response,
    validate_consolidation_response,
    validate_categorization_response,
    validate_synthesis_response,
    ValidationResult,
)


# Paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESURFACE_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# =============================================================================
# REAL RESURFACE DATA LOADING
# =============================================================================

def load_resurface_manifest() -> list[dict]:
    """
    Load the Resurface conversation manifest.

    Returns list of conversation metadata dicts with:
    - id, title, created, message_count, turn_count, source
    """
    manifest_path = RESURFACE_DATA_DIR / "parsed" / "_manifest.json"
    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        data = json.load(f)

    return data.get("conversations", [])


def load_resurface_conversation(conv_id: str) -> dict:
    """Load a specific parsed conversation by ID."""
    conv_path = RESURFACE_DATA_DIR / "parsed" / f"{conv_id}.json"
    if not conv_path.exists():
        return {"error": f"Conversation not found: {conv_id}"}

    with open(conv_path) as f:
        return json.load(f)


def load_resurface_consolidated() -> dict:
    """Load real consolidated data from Resurface."""
    path = RESURFACE_DATA_DIR / "consolidated" / "consolidated.json"
    if not path.exists():
        return {"idea_clusters": [], "problem_clusters": [], "workflow_clusters": []}

    with open(path) as f:
        return json.load(f)


def load_resurface_categorized() -> dict:
    """Load real categorized data from Resurface."""
    path = RESURFACE_DATA_DIR / "consolidated" / "categorized.json"
    if not path.exists():
        return {"ideas": []}

    with open(path) as f:
        return json.load(f)


def load_resurface_synthesized() -> dict:
    """Load real passion profile from Resurface."""
    path = RESURFACE_DATA_DIR / "synthesized" / "passion_profile.json"
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def get_conversation_choices() -> list[tuple[str, str]]:
    """Get conversation choices for dropdown (label, value)."""
    manifest = load_resurface_manifest()
    choices = []
    for conv in manifest[:100]:  # Limit to first 100 for UI performance
        date = conv.get("created", "")[:10] if conv.get("created") else "?"
        title = conv.get("title", "Untitled")[:50]
        msg_count = conv.get("message_count", 0)
        label = f"[{date}] {title} ({msg_count} msgs)"
        choices.append((label, conv.get("id", "")))
    return choices


def has_real_data() -> bool:
    """Check if Resurface data directory exists with data."""
    manifest_path = RESURFACE_DATA_DIR / "parsed" / "_manifest.json"
    return manifest_path.exists()


# =============================================================================
# SAMPLE DATA LOADING
# =============================================================================

def load_test_conversation(name: str) -> dict:
    """Load a test conversation by name."""
    path = TEST_DATA_DIR / f"{name}.json"
    if not path.exists():
        return {"error": f"File not found: {path}"}
    with open(path) as f:
        return json.load(f)


def load_consolidation_sample() -> dict:
    """Load consolidation sample data."""
    path = TEST_DATA_DIR / "consolidation_sample.json"
    if not path.exists():
        return {"ideas": [], "problems": []}
    with open(path) as f:
        return json.load(f)


def load_categorization_sample() -> dict:
    """Load categorization sample data."""
    path = TEST_DATA_DIR / "categorization_sample.json"
    if not path.exists():
        return {"ideas": []}
    with open(path) as f:
        return json.load(f)


def load_synthesis_sample() -> dict:
    """Load synthesis sample data."""
    path = TEST_DATA_DIR / "synthesis_sample.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def format_conversation_for_prompt(conv: dict) -> str:
    """Format a conversation dict for the extraction prompt."""
    lines = []
    lines.append(f"Title: {conv.get('title', 'Untitled')}")
    lines.append(f"Date: {conv.get('created', 'Unknown')[:10] if conv.get('created') else 'Unknown'}")
    lines.append("")

    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_extraction_test(
    model: str,
    conversation_name: str,
    temperature: float
) -> tuple[str, str, dict]:
    """
    Run extraction test on a sample conversation.

    Returns:
        Tuple of (raw_response, validation_summary, metrics_dict)
    """
    # Load conversation
    conv = load_test_conversation(conversation_name)
    if "error" in conv:
        return "", f"Error: {conv['error']}", {}

    # Build prompt
    conv_text = format_conversation_for_prompt(conv)
    prompt = build_extraction_prompt(conv_text)
    system_prompt = SYSTEM_PROMPTS["extraction"]

    # Call model
    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=ExtractionResult,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    # Validate
    result = validate_extraction_response(response, response_time_ms)

    # Format summary
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def run_real_extraction_test(
    model: str,
    conv_id: str,
    temperature: float
) -> tuple[str, str, dict, str]:
    """
    Run extraction test on a real Resurface conversation.

    Returns:
        Tuple of (raw_response, validation_summary, metrics_dict, conv_title)
    """
    # Load real conversation
    conv = load_resurface_conversation(conv_id)
    if "error" in conv:
        return "", f"Error: {conv['error']}", {}, ""

    conv_title = conv.get("title", "Untitled")

    # Build prompt
    conv_text = format_conversation_for_prompt(conv)
    prompt = build_extraction_prompt(conv_text)
    system_prompt = SYSTEM_PROMPTS["extraction"]

    # Call model
    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=ExtractionResult,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}, conv_title

    # Validate
    result = validate_extraction_response(response, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict(), conv_title


def run_real_consolidation_test(
    model: str,
    item_type: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run consolidation test on real Resurface data."""
    consolidated = load_resurface_consolidated()

    # Map item type to correct key
    key_map = {
        "ideas": "idea_clusters",
        "problems": "problem_clusters",
        "workflows": "workflow_clusters",
    }
    items = consolidated.get(key_map.get(item_type, item_type), [])

    if not items:
        return "", f"No {item_type} found in real data", {}

    # Take a sample for testing (first 10 items)
    items = items[:10]

    prompt = build_consolidation_prompt(item_type, items)
    system_prompt = SYSTEM_PROMPTS["consolidation"]

    schema_map = {
        "ideas": ConsolidatedIdea,
        "problems": ConsolidatedProblem,
    }

    # Use array schema for array output
    item_schema = schema_map.get(item_type)
    array_schema = make_array_schema(item_schema) if item_schema else None

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=array_schema,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_consolidation_response(response, item_type, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def run_real_categorization_test(
    model: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run categorization test on real Resurface data."""
    categorized = load_resurface_categorized()
    ideas = categorized.get("ideas", [])

    if not ideas:
        return "", "No ideas found in real data", {}

    # Take a sample (first 10 ideas)
    ideas = ideas[:10]

    prompt = build_scoring_prompt(ideas)
    system_prompt = SYSTEM_PROMPTS["categorization"]

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=make_array_schema(ScoredIdea),
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_categorization_response(response, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def run_real_synthesis_test(
    model: str,
    synthesis_type: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run synthesis test on real Resurface data."""
    profile = load_resurface_synthesized()
    consolidated = load_resurface_consolidated()

    if not profile:
        return "", "No passion profile found. Run synthesis in Resurface first.", {}

    if synthesis_type == "intersection":
        # Extract themes from profile
        themes = [t.get("theme", "") for t in profile.get("core_themes", [])[:8]]
        tools = [t.get("tool", "") for t in profile.get("tool_expertise", [])[:15]]
        profile_summary = profile.get("summary", "A developer with various interests.")

        prompt = build_intersection_prompt(profile_summary, themes, tools)
        system_prompt = SYSTEM_PROMPTS["synthesis_intersection"]
        schema = make_array_schema(IntersectionIdea)
    else:  # solution
        problems = profile.get("recurring_problems", [])[:10]
        tools = [t.get("tool", "") for t in profile.get("tool_expertise", [])[:15]]
        profile_summary = profile.get("summary", "A developer with various interests.")

        prompt = build_solution_prompt(profile_summary, problems, tools)
        system_prompt = SYSTEM_PROMPTS["synthesis_solution"]
        schema = make_array_schema(SolutionIdea)

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_synthesis_response(response, synthesis_type, response_time_ms)
    validation_summary = format_validation_summary(result)

    return response, validation_summary, result.to_dict()


def run_consolidation_test(
    model: str,
    item_type: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run consolidation test."""
    sample = load_consolidation_sample()
    items = sample.get(item_type, [])

    if not items:
        return "", f"No {item_type} found in sample data", {}

    prompt = build_consolidation_prompt(item_type, items)
    system_prompt = SYSTEM_PROMPTS["consolidation"]

    schema_map = {
        "ideas": ConsolidatedIdea,
        "problems": ConsolidatedProblem,
    }

    # Use array schema for array output
    item_schema = schema_map.get(item_type)
    array_schema = make_array_schema(item_schema) if item_schema else None

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=array_schema,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_consolidation_response(response, item_type, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def run_categorization_test(
    model: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run categorization/scoring test."""
    sample = load_categorization_sample()
    ideas = sample.get("ideas", [])

    if not ideas:
        return "", "No ideas found in sample data", {}

    prompt = build_scoring_prompt(ideas)
    system_prompt = SYSTEM_PROMPTS["categorization"]

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=make_array_schema(ScoredIdea),
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_categorization_response(response, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def run_synthesis_test(
    model: str,
    synthesis_type: str,
    temperature: float
) -> tuple[str, str, dict]:
    """Run synthesis test (intersection or solution ideas)."""
    sample = load_synthesis_sample()

    if synthesis_type == "intersection":
        prompt = build_intersection_prompt(
            sample.get("profile_summary", "A developer interested in various topics."),
            sample.get("themes", []),
            sample.get("tools", [])
        )
        system_prompt = SYSTEM_PROMPTS["synthesis_intersection"]
        schema = make_array_schema(IntersectionIdea)
    else:  # solution
        prompt = build_solution_prompt(
            sample.get("profile_summary", "A developer interested in various topics."),
            sample.get("problems", []),
            sample.get("tools", [])
        )
        system_prompt = SYSTEM_PROMPTS["synthesis_solution"]
        schema = make_array_schema(SolutionIdea)

    try:
        response, duration = chat_with_system(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema=schema,
            temperature=temperature
        )
        response_time_ms = int(duration * 1000)
    except (OllamaConnectionError, OllamaError) as e:
        return "", f"Error: {str(e)}", {}

    result = validate_synthesis_response(response, synthesis_type, response_time_ms)
    summary = format_validation_summary(result)

    return response, summary, result.to_dict()


def format_validation_summary(result: ValidationResult) -> str:
    """Format validation result as readable summary."""
    lines = []

    # Status indicators
    json_icon = "✓" if result.json_parsed else "✗"
    schema_icon = "✓" if result.schema_valid else "✗"
    fields_icon = "✓" if result.required_fields_present else "✗"
    passed_icon = "✓" if result.passed else "✗"

    lines.append(f"## Validation Results\n")
    lines.append(f"**Overall**: {passed_icon} {'PASSED' if result.passed else 'FAILED'}")
    lines.append(f"**Score**: {result.score:.0%}\n")

    lines.append("### Checks")
    lines.append(f"- {json_icon} JSON Parsed")
    lines.append(f"- {schema_icon} Schema Valid")
    lines.append(f"- {fields_icon} Required Fields Present")
    lines.append(f"\n**Response Time**: {result.response_time_ms}ms")

    if result.parse_error:
        lines.append(f"\n### Parse Error\n{result.parse_error}")

    if result.schema_errors:
        lines.append(f"\n### Schema Errors")
        for err in result.schema_errors[:5]:  # Limit to 5
            lines.append(f"- {err}")
        if len(result.schema_errors) > 5:
            lines.append(f"- ... and {len(result.schema_errors) - 5} more")

    if result.missing_fields:
        lines.append(f"\n### Missing Fields")
        for field in result.missing_fields[:5]:
            lines.append(f"- {field}")

    if result.value_errors:
        lines.append(f"\n### Value Errors")
        for err in result.value_errors[:5]:
            lines.append(f"- {err}")

    return "\n".join(lines)


# =============================================================================
# BATCH TESTING
# =============================================================================

TEST_SUITE = [
    ("extraction_simple", "Extraction", "sample_simple"),
    ("extraction_medium", "Extraction", "sample_medium"),
    ("extraction_complex", "Extraction", "sample_complex"),
    ("consolidation_ideas", "Consolidation", "ideas"),
    ("consolidation_problems", "Consolidation", "problems"),
    ("categorization", "Categorization", None),
    ("synthesis_intersection", "Synthesis", "intersection"),
    ("synthesis_solution", "Synthesis", "solution"),
]


def run_batch_tests(
    model: str,
    temperature: float,
    progress: gr.Progress = gr.Progress()
) -> tuple[str, str]:
    """
    Run all tests and return results table.

    Returns:
        Tuple of (markdown_table, json_results)
    """
    results = []

    for i, (test_id, test_type, param) in enumerate(TEST_SUITE):
        progress((i + 1) / len(TEST_SUITE), desc=f"Running {test_id}...")

        try:
            if test_type == "Extraction":
                _, _, metrics = run_extraction_test(model, param, temperature)
            elif test_type == "Consolidation":
                _, _, metrics = run_consolidation_test(model, param, temperature)
            elif test_type == "Categorization":
                _, _, metrics = run_categorization_test(model, temperature)
            elif test_type == "Synthesis":
                _, _, metrics = run_synthesis_test(model, param, temperature)
            else:
                metrics = {"error": "Unknown test type"}

            results.append({
                "test_id": test_id,
                "test_type": test_type,
                **metrics
            })
        except Exception as e:
            results.append({
                "test_id": test_id,
                "test_type": test_type,
                "error": str(e)
            })

    # Generate markdown table
    table_lines = [
        "| Test | JSON | Schema | Fields | Time | Status |",
        "|------|------|--------|--------|------|--------|",
    ]

    pass_count = 0
    for r in results:
        if "error" in r and r.get("error"):
            table_lines.append(f"| {r['test_id']} | - | - | - | - | Error |")
        else:
            json_ok = "✓" if r.get("json_parsed") else "✗"
            schema_ok = "✓" if r.get("schema_valid") else "✗"
            fields_ok = "✓" if r.get("required_fields_present") else "✗"
            time_ms = r.get("response_time_ms", 0)
            passed = r.get("passed", False)
            status = "✓ Pass" if passed else "✗ Fail"
            if passed:
                pass_count += 1
            table_lines.append(
                f"| {r['test_id']} | {json_ok} | {schema_ok} | {fields_ok} | {time_ms}ms | {status} |"
            )

    # Summary
    total = len(results)
    summary = f"\n\n**Summary**: {pass_count}/{total} tests passed ({pass_count/total:.0%})"
    table_lines.append(summary)

    # Recommendation
    if pass_count == total:
        table_lines.append("\n**Recommendation**: This model is Resurface-ready!")
    elif pass_count / total >= 0.85:
        table_lines.append("\n**Recommendation**: Good performance, minor issues to investigate.")
    elif pass_count / total >= 0.5:
        table_lines.append("\n**Recommendation**: Moderate performance, may need prompt tuning.")
    else:
        table_lines.append("\n**Recommendation**: Not recommended for Resurface use.")

    return "\n".join(table_lines), json.dumps(results, indent=2)


def save_batch_results(model: str, results_json: str) -> str:
    """Save batch results to file."""
    if not results_json:
        return "No results to save"

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "_").replace("/", "_")
    filename = f"results_{model_safe}_{timestamp}.json"
    path = RESULTS_DIR / filename

    with open(path, "w") as f:
        f.write(results_json)

    return f"Saved to {path}"


# =============================================================================
# GRADIO UI
# =============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    # Check Ollama connection on startup
    ollama_connected = check_connection()
    initial_models = get_model_names() if ollama_connected else []

    with gr.Blocks(title="Ollama Tester for Resurface") as app:
        gr.Markdown("# Ollama LLM Tester for Resurface")
        gr.Markdown(
            "Test local Ollama models against Resurface's structured output requirements "
            "before integrating them into the main application."
        )

        # Connection status
        if not ollama_connected:
            gr.Markdown(
                "**⚠️ Warning**: Cannot connect to Ollama. "
                "Make sure Ollama is running (`ollama serve`)."
            )

        # =================================================================
        # TAB 1: MODEL SELECTION
        # =================================================================
        with gr.Tab("Model Selection"):
            gr.Markdown("## Select Ollama Model")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=initial_models[0] if initial_models else None,
                    label="Model",
                    info="Select an Ollama model to test"
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=0)

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature",
                    info="0.0 for deterministic output (recommended for testing)"
                )

            connection_status = gr.Markdown(
                "✓ Connected to Ollama" if ollama_connected else "✗ Not connected"
            )

            def refresh_models():
                if check_connection():
                    models = get_model_names()
                    return (
                        gr.update(choices=models, value=models[0] if models else None),
                        "✓ Connected to Ollama"
                    )
                else:
                    return gr.update(choices=[], value=None), "✗ Not connected"

            refresh_btn.click(
                fn=refresh_models,
                outputs=[model_dropdown, connection_status]
            )

        # =================================================================
        # TAB 2: INDIVIDUAL TESTS
        # =================================================================
        with gr.Tab("Individual Tests"):
            gr.Markdown("## Run Individual Tests")

            with gr.Row():
                test_type = gr.Dropdown(
                    choices=["Extraction", "Consolidation", "Categorization", "Synthesis"],
                    value="Extraction",
                    label="Test Type"
                )
                test_param = gr.Dropdown(
                    choices=["sample_simple", "sample_medium", "sample_complex"],
                    value="sample_simple",
                    label="Sample Data"
                )

            run_test_btn = gr.Button("▶️ Run Test", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Raw Response")
                    raw_response = gr.Code(language="json", label="LLM Output")
                with gr.Column():
                    gr.Markdown("### Validation")
                    validation_output = gr.Markdown()

            # Update param choices based on test type
            def update_params(test_type):
                if test_type == "Extraction":
                    return gr.update(
                        choices=["sample_simple", "sample_medium", "sample_complex"],
                        value="sample_simple",
                        visible=True
                    )
                elif test_type == "Consolidation":
                    return gr.update(
                        choices=["ideas", "problems"],
                        value="ideas",
                        visible=True
                    )
                elif test_type == "Categorization":
                    return gr.update(visible=False)
                elif test_type == "Synthesis":
                    return gr.update(
                        choices=["intersection", "solution"],
                        value="intersection",
                        visible=True
                    )
                return gr.update()

            test_type.change(
                fn=update_params,
                inputs=[test_type],
                outputs=[test_param]
            )

            def run_single_test(model, test_type, param, temp):
                if not model:
                    return "", "Please select a model first"

                if test_type == "Extraction":
                    return run_extraction_test(model, param, temp)[:2]
                elif test_type == "Consolidation":
                    return run_consolidation_test(model, param, temp)[:2]
                elif test_type == "Categorization":
                    return run_categorization_test(model, temp)[:2]
                elif test_type == "Synthesis":
                    return run_synthesis_test(model, param, temp)[:2]
                return "", "Unknown test type"

            run_test_btn.click(
                fn=run_single_test,
                inputs=[model_dropdown, test_type, test_param, temperature],
                outputs=[raw_response, validation_output]
            )

        # =================================================================
        # TAB 3: BATCH TESTING
        # =================================================================
        with gr.Tab("Batch Testing"):
            gr.Markdown("## Run All Tests")
            gr.Markdown(
                "Run the complete test suite to evaluate model reliability. "
                "This tests extraction, consolidation, categorization, and synthesis."
            )

            run_batch_btn = gr.Button("▶️ Run All Tests", variant="primary")

            batch_results = gr.Markdown(label="Results")

            with gr.Accordion("Raw JSON Results", open=False):
                json_results = gr.Code(language="json", label="JSON")
                save_btn = gr.Button("💾 Save Results")
                save_status = gr.Textbox(label="Save Status", interactive=False)

            def run_batch(model, temp, progress=gr.Progress()):
                if not model:
                    return "Please select a model first", ""
                return run_batch_tests(model, temp, progress)

            run_batch_btn.click(
                fn=run_batch,
                inputs=[model_dropdown, temperature],
                outputs=[batch_results, json_results]
            )

            save_btn.click(
                fn=save_batch_results,
                inputs=[model_dropdown, json_results],
                outputs=[save_status]
            )

        # =================================================================
        # TAB 4: MODEL COMPARISON
        # =================================================================
        with gr.Tab("Model Comparison"):
            gr.Markdown("## Compare Multiple Models")
            gr.Markdown("Select 2+ models to compare their performance side-by-side.")

            model_select = gr.CheckboxGroup(
                choices=initial_models,
                label="Select Models to Compare"
            )

            compare_btn = gr.Button("▶️ Compare Models", variant="primary")
            comparison_results = gr.Markdown()

            def compare_models(models, temp, progress=gr.Progress()):
                if len(models) < 2:
                    return "Please select at least 2 models to compare"

                all_results = {}
                total_steps = len(models) * len(TEST_SUITE)
                step = 0

                for model in models:
                    model_results = []
                    for test_id, test_type, param in TEST_SUITE:
                        progress(step / total_steps, desc=f"{model}: {test_id}")
                        step += 1

                        try:
                            if test_type == "Extraction":
                                _, _, metrics = run_extraction_test(model, param, temp)
                            elif test_type == "Consolidation":
                                _, _, metrics = run_consolidation_test(model, param, temp)
                            elif test_type == "Categorization":
                                _, _, metrics = run_categorization_test(model, temp)
                            elif test_type == "Synthesis":
                                _, _, metrics = run_synthesis_test(model, param, temp)
                            else:
                                metrics = {}

                            model_results.append(metrics.get("passed", False))
                        except Exception:
                            model_results.append(False)

                    all_results[model] = model_results

                # Build comparison table
                lines = ["| Test |"]
                lines[0] += " | ".join(models) + " |"

                lines.append("|------|" + "|".join(["------"] * len(models)) + "|")

                for i, (test_id, _, _) in enumerate(TEST_SUITE):
                    row = f"| {test_id} |"
                    for model in models:
                        passed = all_results[model][i]
                        row += " ✓ |" if passed else " ✗ |"
                    lines.append(row)

                # Summary row
                lines.append("|------|" + "|".join(["------"] * len(models)) + "|")
                summary_row = "| **Pass Rate** |"
                best_model = None
                best_rate = 0
                for model in models:
                    passed = sum(all_results[model])
                    total = len(TEST_SUITE)
                    rate = passed / total
                    summary_row += f" {passed}/{total} ({rate:.0%}) |"
                    if rate > best_rate:
                        best_rate = rate
                        best_model = model
                lines.append(summary_row)

                # Recommendation
                if best_model and best_rate >= 0.85:
                    lines.append(f"\n**Recommendation**: {best_model} performs best and is suitable for Resurface.")
                elif best_model:
                    lines.append(f"\n**Recommendation**: {best_model} performs best but may need tuning.")

                return "\n".join(lines)

            compare_btn.click(
                fn=compare_models,
                inputs=[model_select, temperature],
                outputs=[comparison_results]
            )

            # Update checkbox choices when models refresh
            def update_comparison_choices():
                if check_connection():
                    models = get_model_names()
                    return gr.update(choices=models)
                return gr.update(choices=[])

        # =================================================================
        # TAB 5: REAL DATA TESTING
        # =================================================================
        with gr.Tab("Real Data Testing"):
            real_data_available = has_real_data()

            if not real_data_available:
                gr.Markdown(
                    "## Real Data Not Available\n\n"
                    "No Resurface data found. To use real data testing:\n"
                    "1. Run Resurface and import your conversations\n"
                    "2. Run extraction, consolidation, and synthesis\n"
                    "3. Return here to test with real data"
                )
            else:
                gr.Markdown("## Test with Real Resurface Data")
                gr.Markdown(
                    "Test Ollama models against your actual GPT/Claude conversation exports. "
                    "This validates model performance on real-world data."
                )

                # Get conversation choices
                conv_choices = get_conversation_choices()
                manifest = load_resurface_manifest()

                gr.Markdown(f"**{len(manifest)} conversations available** (showing first 100)")

                with gr.Row():
                    real_test_type = gr.Dropdown(
                        choices=["Extraction", "Consolidation", "Categorization", "Synthesis"],
                        value="Extraction",
                        label="Test Type"
                    )
                    real_test_param = gr.Dropdown(
                        choices=conv_choices,
                        value=conv_choices[0][1] if conv_choices else None,
                        label="Select Conversation"
                    )

                run_real_test_btn = gr.Button("▶️ Run Test on Real Data", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Raw Response")
                        real_raw_response = gr.Code(language="json", label="LLM Output")
                    with gr.Column():
                        gr.Markdown("### Validation")
                        real_validation_output = gr.Markdown()

                # Update param choices based on test type
                def update_real_params(test_type):
                    if test_type == "Extraction":
                        choices = get_conversation_choices()
                        return gr.update(
                            choices=choices,
                            value=choices[0][1] if choices else None,
                            label="Select Conversation",
                            visible=True
                        )
                    elif test_type == "Consolidation":
                        return gr.update(
                            choices=[("Ideas", "ideas"), ("Problems", "problems"), ("Workflows", "workflows")],
                            value="ideas",
                            label="Item Type",
                            visible=True
                        )
                    elif test_type == "Categorization":
                        return gr.update(visible=False)
                    elif test_type == "Synthesis":
                        return gr.update(
                            choices=[("Intersection Ideas", "intersection"), ("Solution Ideas", "solution")],
                            value="intersection",
                            label="Synthesis Type",
                            visible=True
                        )
                    return gr.update()

                real_test_type.change(
                    fn=update_real_params,
                    inputs=[real_test_type],
                    outputs=[real_test_param]
                )

                def run_real_single_test(model, test_type, param, temp):
                    if not model:
                        return "", "Please select a model first"

                    if test_type == "Extraction":
                        resp, summary, _, title = run_real_extraction_test(model, param, temp)
                        return resp, f"**Conversation**: {title}\n\n{summary}"
                    elif test_type == "Consolidation":
                        return run_real_consolidation_test(model, param, temp)[:2]
                    elif test_type == "Categorization":
                        return run_real_categorization_test(model, temp)[:2]
                    elif test_type == "Synthesis":
                        return run_real_synthesis_test(model, param, temp)[:2]
                    return "", "Unknown test type"

                run_real_test_btn.click(
                    fn=run_real_single_test,
                    inputs=[model_dropdown, real_test_type, real_test_param, temperature],
                    outputs=[real_raw_response, real_validation_output]
                )

                # Batch testing on random real conversations
                gr.Markdown("---")
                gr.Markdown("### Batch Test on Random Real Conversations")

                with gr.Row():
                    num_random_convs = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of conversations to test"
                    )
                    run_random_batch_btn = gr.Button("▶️ Run Random Batch", variant="secondary")

                random_batch_results = gr.Markdown()

                def run_random_real_batch(model, temp, num_convs, progress=gr.Progress()):
                    import random

                    if not model:
                        return "Please select a model first"

                    manifest = load_resurface_manifest()
                    if len(manifest) < num_convs:
                        num_convs = len(manifest)

                    # Select random conversations
                    selected = random.sample(manifest, int(num_convs))

                    results = []
                    for i, conv_meta in enumerate(selected):
                        progress((i + 1) / num_convs, desc=f"Testing: {conv_meta.get('title', 'Untitled')[:30]}...")

                        conv_id = conv_meta.get("id")
                        _, _, metrics, title = run_real_extraction_test(model, conv_id, temp)

                        results.append({
                            "title": title[:40],
                            "passed": metrics.get("passed", False),
                            "json_parsed": metrics.get("json_parsed", False),
                            "schema_valid": metrics.get("schema_valid", False),
                            "time_ms": metrics.get("response_time_ms", 0)
                        })

                    # Build results table
                    lines = [
                        "| Conversation | JSON | Schema | Time | Status |",
                        "|--------------|------|--------|------|--------|",
                    ]
                    pass_count = 0
                    for r in results:
                        json_ok = "✓" if r["json_parsed"] else "✗"
                        schema_ok = "✓" if r["schema_valid"] else "✗"
                        status = "✓ Pass" if r["passed"] else "✗ Fail"
                        if r["passed"]:
                            pass_count += 1
                        lines.append(
                            f"| {r['title']} | {json_ok} | {schema_ok} | {r['time_ms']}ms | {status} |"
                        )

                    total = len(results)
                    lines.append(f"\n**Summary**: {pass_count}/{total} passed ({pass_count/total:.0%})")

                    return "\n".join(lines)

                run_random_batch_btn.click(
                    fn=run_random_real_batch,
                    inputs=[model_dropdown, temperature, num_random_convs],
                    outputs=[random_batch_results]
                )

        # =================================================================
        # TAB 6: SAMPLE DATA
        # =================================================================
        with gr.Tab("Sample Data"):
            gr.Markdown("## View Test Data")
            gr.Markdown("Browse the sample conversations and data used for testing.")

            data_selector = gr.Dropdown(
                choices=[
                    "sample_simple",
                    "sample_medium",
                    "sample_complex",
                    "consolidation_sample",
                    "categorization_sample",
                    "synthesis_sample"
                ],
                value="sample_simple",
                label="Select Data"
            )

            data_display = gr.Code(language="json", label="Data")

            def load_data(name):
                path = TEST_DATA_DIR / f"{name}.json"
                if path.exists():
                    with open(path) as f:
                        return json.dumps(json.load(f), indent=2)
                return f"File not found: {path}"

            data_selector.change(
                fn=load_data,
                inputs=[data_selector],
                outputs=[data_display]
            )

            # Load initial data
            app.load(
                fn=lambda: load_data("sample_simple"),
                outputs=[data_display]
            )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )
