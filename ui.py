#!/usr/bin/env python3
"""
Resurface UI - Gradio interface for conversation extraction and analysis.
"""
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Paths
PARSED_DIR = Path("data/parsed")
EXTRACTIONS_DIR = Path("data/extractions")
MANIFEST_FILE = PARSED_DIR / "_manifest.json"

# =============================================================================
# MATRIX THEME
# =============================================================================

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

# Custom green color palette for matrix theme
class MatrixGreen(colors.Color):
    pass

matrix_green = MatrixGreen(
    c50="#001a00",
    c100="#003300",
    c200="#004d00",
    c300="#006600",
    c400="#008000",
    c500="#00ff00",
    c600="#00ff00",
    c700="#00ff00",
    c800="#00ff00",
    c900="#00ff00",
    c950="#00ff00",
)

class MatrixTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=matrix_green,
            secondary_hue=matrix_green,
            neutral_hue=matrix_green,
            font=fonts.GoogleFont("Source Code Pro"),
            font_mono=fonts.GoogleFont("Source Code Pro"),
        )
        # Override all colors to pure black/green
        super().set(
            # Body/background
            body_background_fill="#000000",
            body_background_fill_dark="#000000",
            body_text_color="#00ff00",
            body_text_color_dark="#00ff00",
            body_text_color_subdued="#006600",
            body_text_color_subdued_dark="#006600",

            # Blocks
            block_background_fill="#000000",
            block_background_fill_dark="#000000",
            block_border_color="#00ff00",
            block_border_color_dark="#00ff00",
            block_label_background_fill="#000000",
            block_label_background_fill_dark="#000000",
            block_label_text_color="#00ff00",
            block_label_text_color_dark="#00ff00",
            block_title_text_color="#00ff00",
            block_title_text_color_dark="#00ff00",

            # Panels
            panel_background_fill="#000000",
            panel_background_fill_dark="#000000",
            panel_border_color="#00ff00",
            panel_border_color_dark="#00ff00",

            # Buttons
            button_primary_background_fill="#003300",
            button_primary_background_fill_dark="#003300",
            button_primary_background_fill_hover="#004d00",
            button_primary_background_fill_hover_dark="#004d00",
            button_primary_text_color="#00ff00",
            button_primary_text_color_dark="#00ff00",
            button_primary_border_color="#00ff00",
            button_primary_border_color_dark="#00ff00",
            button_secondary_background_fill="#000000",
            button_secondary_background_fill_dark="#000000",
            button_secondary_background_fill_hover="#001a00",
            button_secondary_background_fill_hover_dark="#001a00",
            button_secondary_text_color="#00ff00",
            button_secondary_text_color_dark="#00ff00",
            button_secondary_border_color="#00ff00",
            button_secondary_border_color_dark="#00ff00",
            button_cancel_background_fill="#000000",
            button_cancel_background_fill_dark="#000000",
            button_cancel_text_color="#00ff00",
            button_cancel_text_color_dark="#00ff00",

            # Inputs
            input_background_fill="#000000",
            input_background_fill_dark="#000000",
            input_border_color="#00ff00",
            input_border_color_dark="#00ff00",
            input_border_color_focus="#00ff00",
            input_border_color_focus_dark="#00ff00",
            input_placeholder_color="#006600",
            input_placeholder_color_dark="#006600",

            # Tables
            table_border_color="#00ff00",
            table_border_color_dark="#00ff00",
            table_even_background_fill="#000000",
            table_even_background_fill_dark="#000000",
            table_odd_background_fill="#001a00",
            table_odd_background_fill_dark="#001a00",
            table_row_focus="#003300",
            table_row_focus_dark="#003300",

            # Checkboxes
            checkbox_background_color="#000000",
            checkbox_background_color_dark="#000000",
            checkbox_background_color_selected="#00ff00",
            checkbox_background_color_selected_dark="#00ff00",
            checkbox_border_color="#00ff00",
            checkbox_border_color_dark="#00ff00",
            checkbox_label_text_color="#00ff00",
            checkbox_label_text_color_dark="#00ff00",

            # Sliders
            slider_color="#00ff00",
            slider_color_dark="#00ff00",

            # Links
            link_text_color="#00ff00",
            link_text_color_dark="#00ff00",
            link_text_color_hover="#00cc00",
            link_text_color_hover_dark="#00cc00",
            link_text_color_visited="#009900",
            link_text_color_visited_dark="#009900",
            link_text_color_active="#00ff00",
            link_text_color_active_dark="#00ff00",

            # Borders
            border_color_accent="#00ff00",
            border_color_accent_dark="#00ff00",
            border_color_primary="#00ff00",
            border_color_primary_dark="#00ff00",

            # Shadows - remove them for flat look
            shadow_drop="none",
            shadow_drop_lg="none",
            shadow_inset="none",
            shadow_spread="0px",
            shadow_spread_dark="0px",

            # Background colors
            background_fill_primary="#000000",
            background_fill_primary_dark="#000000",
            background_fill_secondary="#000000",
            background_fill_secondary_dark="#000000",

            # Color accents
            color_accent="#00ff00",
            color_accent_soft="#003300",
            color_accent_soft_dark="#003300",
        )

# Additional CSS for elements theme doesn't cover
MATRIX_CSS = """
/* Force dark mode and remove any light backgrounds */
:root {
    color-scheme: dark !important;
}

/* Remove Gradio footer */
footer {
    display: none !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    background: #000000;
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #00ff00;
    border-radius: 4px;
}
::-webkit-scrollbar-track {
    background: #001a00;
}

/* Tab styling */
.tab-nav button {
    background: #000000 !important;
    color: #00ff00 !important;
    border: 1px solid #00ff00 !important;
}
.tab-nav button.selected {
    background: #003300 !important;
}

/* Tab underline - remove orange */
.tab-nav button.selected::after,
button.tab-nav-button.selected::after,
.tab-nav button::after,
button.tab-nav-button::after {
    background: #00ff00 !important;
    background-color: #00ff00 !important;
    border-bottom-color: #00ff00 !important;
}
.tabs .tabitem {
    border-color: #00ff00 !important;
}
/* More aggressive tab underline override */
[role="tablist"] button[aria-selected="true"]::after,
[role="tablist"] button.selected::after,
.tab-nav .selected::after,
.svelte-1kcgrqr::after,
button[role="tab"][aria-selected="true"]::after {
    background: #00ff00 !important;
    background-color: #00ff00 !important;
}
/* Target any orange color directly */
[style*="rgb(249, 115, 22)"],
[style*="#f97316"],
[style*="orange"] {
    background: #00ff00 !important;
    background-color: #00ff00 !important;
    border-color: #00ff00 !important;
}

/* Primary buttons - force green styling */
button.primary,
button.lg.primary,
button[variant="primary"],
.primary-btn,
.gr-button-primary {
    background: #003300 !important;
    background-color: #003300 !important;
    color: #00ff00 !important;
    border: 1px solid #00ff00 !important;
}
button.primary:hover,
button.lg.primary:hover {
    background: #004400 !important;
    background-color: #004400 !important;
}

/* Slider track - green instead of orange */
input[type="range"] {
    accent-color: #00ff00 !important;
}
input[type="range"]::-webkit-slider-runnable-track {
    background: linear-gradient(to right, #00ff00 0%, #00ff00 var(--value-percent, 50%), #003300 var(--value-percent, 50%), #003300 100%) !important;
}
input[type="range"]::-moz-range-track {
    background: #003300 !important;
}
input[type="range"]::-moz-range-progress {
    background: #00ff00 !important;
}
.range-slider,
.slider {
    --slider-color: #00ff00 !important;
}

/* Radio buttons - green instead of orange */
input[type="radio"] {
    accent-color: #00ff00 !important;
}
input[type="radio"]:checked {
    background-color: #00ff00 !important;
    border-color: #00ff00 !important;
}
.gr-radio-row input[type="radio"]:checked + label::before {
    background-color: #00ff00 !important;
}

/* Ensure all text is green */
* {
    color: #00ff00;
}

/* SVG icons */
svg path, svg circle, svg rect {
    stroke: #00ff00 !important;
}

/* Force all gray backgrounds to pure black */
.block, .form, .panel, .container, .wrap, .wrap-inner,
.gr-box, .gr-panel, .gr-form, .gr-block, .gr-padded,
.gradio-container, .contain, .gap, .gr-group,
[class*="block"], [class*="panel"], [class*="container"],
.svelte-1ed2p3z, .svelte-1kcgrqr, .svelte-1guhx2a,
div[data-testid], main, section, article, aside,
.input-container, .output-container, .component-wrapper {
    background: #000000 !important;
    background-color: #000000 !important;
}

/* Table backgrounds */
table, thead, tbody, tr, th, td,
.table-wrap, .dataframe, .gr-dataframe {
    background: #000000 !important;
    background-color: #000000 !important;
}
table tr:nth-child(even),
table tr:nth-child(odd) {
    background: #000000 !important;
}
table tr:hover {
    background: #001a00 !important;
}

/* Input/textarea backgrounds */
input, textarea, select, .gr-input, .gr-textarea,
.textbox, .gr-textbox, [data-testid="textbox"] {
    background: #000000 !important;
    background-color: #000000 !important;
}

/* Dropdown/select backgrounds */
.dropdown, .gr-dropdown, select, option,
[data-testid="dropdown"], .choices, .choices__inner {
    background: #000000 !important;
    background-color: #000000 !important;
}

/* Any remaining gray shades */
[style*="rgb(31, 41, 55)"],
[style*="rgb(17, 24, 39)"],
[style*="rgb(55, 65, 81)"],
[style*="rgb(75, 85, 99)"],
[style*="rgb(107, 114, 128)"],
[style*="#1f2937"],
[style*="#111827"],
[style*="#374151"],
[style*="#4b5563"],
[style*="#6b7280"] {
    background: #000000 !important;
    background-color: #000000 !important;
}

/* Body and html */
html, body {
    background: #000000 !important;
    background-color: #000000 !important;
}

/* Additional targeted selectors for remaining gray elements */
.gr-box, .gr-panel, .gr-form, .gr-input, .gr-button,
.secondary, .gr-secondary, .svelte-1ipelgc, .svelte-1ed2p3z,
.border, .rounded, .shadow, .bg-gray-50, .bg-gray-100,
.bg-gray-200, .bg-gray-700, .bg-gray-800, .bg-gray-900,
div[class*="svelte-"], span[class*="svelte-"] {
    background-color: #000000 !important;
}

/* Borders should be green */
.gr-box, .gr-panel, .gr-form, .gr-input, .gr-button,
.border, input, textarea, select, button, table, th, td {
    border-color: #00ff00 !important;
}

/* Remove any box shadows that might appear gray */
.gr-box, .gr-panel, .shadow, [class*="shadow"] {
    box-shadow: none !important;
}
"""

# =============================================================================
# PLOTLY MATRIX THEME HELPER
# =============================================================================

def apply_matrix_theme(fig):
    """Apply matrix theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#00ff00', family='Courier New'),
        title_font=dict(color='#00ff00'),
        legend=dict(font=dict(color='#00ff00')),
        xaxis=dict(
            gridcolor='#003300',
            linecolor='#00ff00',
            tickfont=dict(color='#00ff00')
        ),
        yaxis=dict(
            gridcolor='#003300',
            linecolor='#00ff00',
            tickfont=dict(color='#00ff00')
        )
    )
    return fig


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_manifest() -> dict:
    """Load the parsed conversations manifest."""
    if not MANIFEST_FILE.exists():
        return {"conversations": [], "total_conversations": 0}
    with open(MANIFEST_FILE, 'r') as f:
        return json.load(f)


def get_status() -> dict:
    """Get current extraction status."""
    manifest = load_manifest()
    total = len(manifest.get('conversations', []))

    extracted = 0
    errors = 0

    for conv in manifest.get('conversations', []):
        conv_id = conv['id']
        extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"

        if extraction_file.exists():
            extracted += 1
            with open(extraction_file, 'r') as f:
                extraction = json.load(f)
                if extraction.get('extraction', {}).get('error'):
                    errors += 1

    return {
        "total": total,
        "extracted": extracted,
        "remaining": total - extracted,
        "errors": errors,
        "progress": (extracted / total * 100) if total > 0 else 0
    }


def load_all_extractions() -> list[dict]:
    """Load all extraction files."""
    extractions = []
    if not EXTRACTIONS_DIR.exists():
        return extractions

    for f in EXTRACTIONS_DIR.glob("*.json"):
        with open(f, 'r') as file:
            extractions.append(json.load(file))

    return extractions


def load_conversation(conv_id: str) -> dict | None:
    """Load a parsed conversation by ID."""
    conv_file = PARSED_DIR / f"{conv_id}.json"
    if not conv_file.exists():
        return None
    with open(conv_file, 'r') as f:
        return json.load(f)


def load_extraction(conv_id: str) -> dict | None:
    """Load extraction for a conversation."""
    extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"
    if not extraction_file.exists():
        return None
    with open(extraction_file, 'r') as f:
        return json.load(f)


def aggregate_ideas() -> list[dict]:
    """Aggregate all project ideas from extractions."""
    ideas = []
    for ext in load_all_extractions():
        extraction = ext.get('extraction', {})
        if extraction.get('error') or extraction.get('empty'):
            continue

        conv_title = ext.get('conversation_title', 'Untitled')
        conv_date = ext.get('conversation_date', '')
        conv_id = ext.get('conversation_id', '')

        for idea in extraction.get('project_ideas', []):
            ideas.append({
                'idea': idea.get('idea', ''),
                'motivation': idea.get('motivation', ''),
                'detail_level': idea.get('detail_level', 'unknown'),
                'source_title': conv_title,
                'source_date': conv_date,
                'source_id': conv_id
            })

    return ideas


def aggregate_problems() -> list[dict]:
    """Aggregate all problems from extractions."""
    problems = []
    for ext in load_all_extractions():
        extraction = ext.get('extraction', {})
        if extraction.get('error') or extraction.get('empty'):
            continue

        conv_title = ext.get('conversation_title', 'Untitled')
        conv_date = ext.get('conversation_date', '')
        conv_id = ext.get('conversation_id', '')

        for prob in extraction.get('problems', []):
            problems.append({
                'problem': prob.get('problem', ''),
                'context': prob.get('context', ''),
                'source_title': conv_title,
                'source_date': conv_date,
                'source_id': conv_id
            })

    return problems


def aggregate_tools() -> dict[str, int]:
    """Aggregate tool mentions with counts."""
    tools = Counter()
    for ext in load_all_extractions():
        extraction = ext.get('extraction', {})
        if extraction.get('error') or extraction.get('empty'):
            continue

        for tool in extraction.get('tools_explored', []):
            tools[tool] += 1

    return dict(tools)


def aggregate_emotions() -> dict:
    """Aggregate emotional signals."""
    emotions = {'excited': [], 'frustrated': [], 'curious': [], 'stuck': [], 'neutral': []}

    for ext in load_all_extractions():
        extraction = ext.get('extraction', {})
        if extraction.get('error') or extraction.get('empty'):
            continue

        signals = extraction.get('emotional_signals', {})
        tone = signals.get('tone', 'neutral')
        if tone in emotions:
            emotions[tone].append({
                'title': ext.get('conversation_title', 'Untitled'),
                'date': ext.get('conversation_date', ''),
                'notes': signals.get('notes', ''),
                'id': ext.get('conversation_id', '')
            })

    return emotions


# =============================================================================
# DASHBOARD TAB
# =============================================================================

def create_dashboard_tab():
    """Create the dashboard tab."""
    with gr.Tab("Dashboard"):
        gr.Markdown("## Resurface Dashboard")

        with gr.Row():
            total_box = gr.Number(label="Total Conversations", interactive=False)
            extracted_box = gr.Number(label="Extracted", interactive=False)
            remaining_box = gr.Number(label="Remaining", interactive=False)
            errors_box = gr.Number(label="Errors", interactive=False)

        progress_bar = gr.Slider(
            label="Extraction Progress",
            minimum=0,
            maximum=100,
            interactive=False
        )

        gr.Markdown("### Quick Stats")
        with gr.Row():
            ideas_count = gr.Number(label="Total Ideas", interactive=False)
            problems_count = gr.Number(label="Total Problems", interactive=False)
            tools_count = gr.Number(label="Unique Tools", interactive=False)

        gr.Markdown("### Recent Extractions")
        recent_table = gr.Dataframe(
            headers=["Title", "Date", "Ideas", "Problems", "Tone"],
            label="",
            interactive=False
        )

        refresh_btn = gr.Button("Refresh Dashboard", variant="primary")

        def refresh_dashboard():
            status = get_status()
            extractions = load_all_extractions()

            # Quick stats
            ideas = aggregate_ideas()
            problems = aggregate_problems()
            tools = aggregate_tools()

            # Recent extractions table
            recent_data = []
            sorted_extractions = sorted(
                extractions,
                key=lambda x: x.get('extracted_at', ''),
                reverse=True
            )[:10]

            for ext in sorted_extractions:
                extraction = ext.get('extraction', {})
                if extraction.get('error'):
                    tone = "ERROR"
                    n_ideas = 0
                    n_probs = 0
                elif extraction.get('empty'):
                    tone = "empty"
                    n_ideas = 0
                    n_probs = 0
                else:
                    tone = extraction.get('emotional_signals', {}).get('tone', '-')
                    n_ideas = len(extraction.get('project_ideas', []))
                    n_probs = len(extraction.get('problems', []))

                recent_data.append([
                    ext.get('conversation_title', 'Untitled')[:40],
                    ext.get('conversation_date', '-'),
                    n_ideas,
                    n_probs,
                    tone
                ])

            return (
                status['total'],
                status['extracted'],
                status['remaining'],
                status['errors'],
                status['progress'],
                len(ideas),
                len(problems),
                len(tools),
                recent_data
            )

        refresh_btn.click(
            fn=refresh_dashboard,
            outputs=[
                total_box, extracted_box, remaining_box, errors_box,
                progress_bar, ideas_count, problems_count, tools_count,
                recent_table
            ]
        )

    return (refresh_dashboard, [
        total_box, extracted_box, remaining_box, errors_box,
        progress_bar, ideas_count, problems_count, tools_count,
        recent_table
    ])


# =============================================================================
# EXTRACTION CONTROL TAB
# =============================================================================

def create_extraction_tab():
    """Create the extraction control tab."""
    with gr.Tab("Extraction"):
        gr.Markdown("## Extraction Control")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Settings")
                from config import load_config
                config = load_config()
                gr.Markdown(f"**Model:** {config.get('model', 'unknown')}")
                gr.Markdown(f"**Provider:** {config.get('api_provider', 'unknown')}")
                gr.Markdown(f"**Rate limit:** {config.get('requests_per_minute', 20)} req/min")

            with gr.Column():
                gr.Markdown("### Run Extraction")
                count_slider = gr.Slider(
                    label="Number to extract",
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1
                )
                extract_btn = gr.Button("Start Extraction", variant="primary")

        output_log = gr.Textbox(
            label="Extraction Log",
            lines=15,
            max_lines=30,
            interactive=False
        )

        gr.Markdown("### Extract Specific Conversation")
        with gr.Row():
            conv_id_input = gr.Textbox(label="Conversation ID", placeholder="Enter UUID...")
            extract_one_btn = gr.Button("Extract This")

        def run_extraction(count):
            try:
                result = subprocess.run(
                    ["python3", "runner.py", "--count", str(int(count))],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                return result.stdout + result.stderr
            except Exception as e:
                return f"Error: {e}"

        def run_single_extraction(conv_id):
            if not conv_id.strip():
                return "Please enter a conversation ID"
            try:
                result = subprocess.run(
                    ["python3", "runner.py", "--id", conv_id.strip()],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                return result.stdout + result.stderr
            except Exception as e:
                return f"Error: {e}"

        extract_btn.click(fn=run_extraction, inputs=[count_slider], outputs=[output_log])
        extract_one_btn.click(fn=run_single_extraction, inputs=[conv_id_input], outputs=[output_log])


# =============================================================================
# CONVERSATION BROWSER TAB
# =============================================================================

def create_browser_tab():
    """Create the conversation browser tab."""
    with gr.Tab("Conversations"):
        gr.Markdown("## Conversation Browser")

        with gr.Row():
            search_box = gr.Textbox(label="Search titles", placeholder="Type to search...")
            filter_extracted = gr.Checkbox(label="Only show extracted", value=False)

        conversations_table = gr.Dataframe(
            headers=["Title", "Date", "Messages", "Extracted", "ID"],
            label="Conversations",
            interactive=False
        )

        load_btn = gr.Button("Load Conversations")

        gr.Markdown("### Conversation Detail")
        conv_id_display = gr.Textbox(label="Selected ID", interactive=False)

        with gr.Tabs():
            with gr.Tab("Messages"):
                messages_display = gr.Markdown("Select a conversation to view messages")

            with gr.Tab("Extraction"):
                extraction_display = gr.JSON(label="Extraction Data")

        def load_conversations(search_term, only_extracted):
            manifest = load_manifest()
            rows = []

            for conv in manifest.get('conversations', []):
                title = conv.get('title', 'Untitled')

                # Search filter
                if search_term and search_term.lower() not in title.lower():
                    continue

                conv_id = conv['id']
                extracted = (EXTRACTIONS_DIR / f"{conv_id}.json").exists()

                # Extracted filter
                if only_extracted and not extracted:
                    continue

                rows.append([
                    title[:50],
                    conv.get('created', '')[:10],
                    conv.get('message_count', 0),
                    "Yes" if extracted else "No",
                    conv_id
                ])

            return rows[:100]  # Limit to 100 for performance

        def view_conversation(evt: gr.SelectData, data):
            if evt.index[0] is None:
                return "", "Select a conversation", None

            row = data[evt.index[0]]
            conv_id = row[4]

            # Load conversation
            conv = load_conversation(conv_id)
            if not conv:
                return conv_id, "Conversation not found", None

            # Format messages
            messages_md = f"## {conv.get('title', 'Untitled')}\n\n"
            messages_md += f"*{conv.get('created', '')[:10]} | {conv.get('message_count', 0)} messages*\n\n---\n\n"

            for msg in conv.get('messages', []):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')[:2000]  # Truncate long messages
                messages_md += f"**{role}:**\n\n{content}\n\n---\n\n"

            # Load extraction if exists
            extraction = load_extraction(conv_id)

            return conv_id, messages_md, extraction

        load_btn.click(
            fn=load_conversations,
            inputs=[search_box, filter_extracted],
            outputs=[conversations_table]
        )

        conversations_table.select(
            fn=view_conversation,
            inputs=[conversations_table],
            outputs=[conv_id_display, messages_display, extraction_display]
        )


# =============================================================================
# PROJECT IDEAS TAB
# =============================================================================

def create_ideas_tab():
    """Create the project ideas tab."""
    with gr.Tab("Ideas"):
        gr.Markdown("## Project Ideas")

        with gr.Row():
            detail_filter = gr.Radio(
                choices=["All", "vague", "sketched", "detailed"],
                value="All",
                label="Detail Level"
            )
            search_ideas = gr.Textbox(label="Search", placeholder="Search ideas...")

        ideas_table = gr.Dataframe(
            headers=["Idea", "Motivation", "Detail", "Source", "Date"],
            label="",
            interactive=False
        )

        load_ideas_btn = gr.Button("Load Ideas")

        # Stats
        with gr.Row():
            ideas_chart = gr.Plot(label="Ideas by Detail Level")

        def load_ideas(detail_level, search_term):
            ideas = aggregate_ideas()
            rows = []

            for idea in ideas:
                # Detail filter
                if detail_level != "All" and idea['detail_level'] != detail_level:
                    continue

                # Search filter
                if search_term:
                    combined = f"{idea['idea']} {idea['motivation']}".lower()
                    if search_term.lower() not in combined:
                        continue

                rows.append([
                    idea['idea'][:100],
                    idea['motivation'][:100],
                    idea['detail_level'],
                    idea['source_title'][:30],
                    idea['source_date']
                ])

            # Create chart
            if ideas:
                detail_counts = Counter(i['detail_level'] for i in ideas)
                fig = px.pie(
                    values=list(detail_counts.values()),
                    names=list(detail_counts.keys()),
                    title="Ideas by Detail Level",
                    color_discrete_sequence=['#00ff00', '#00cc00', '#009900', '#006600']
                )
                apply_matrix_theme(fig)
            else:
                fig = go.Figure()
                apply_matrix_theme(fig)

            return rows, fig

        load_ideas_btn.click(
            fn=load_ideas,
            inputs=[detail_filter, search_ideas],
            outputs=[ideas_table, ideas_chart]
        )


# =============================================================================
# PROBLEMS TAB
# =============================================================================

def create_problems_tab():
    """Create the problems tab."""
    with gr.Tab("Problems"):
        gr.Markdown("## Problems & Pain Points")

        search_problems = gr.Textbox(label="Search", placeholder="Search problems...")

        problems_table = gr.Dataframe(
            headers=["Problem", "Context", "Source", "Date"],
            label="",
            interactive=False
        )

        load_problems_btn = gr.Button("Load Problems")

        def load_problems(search_term):
            problems = aggregate_problems()
            rows = []

            for prob in problems:
                # Search filter
                if search_term:
                    combined = f"{prob['problem']} {prob['context']}".lower()
                    if search_term.lower() not in combined:
                        continue

                rows.append([
                    prob['problem'][:100],
                    prob['context'][:100],
                    prob['source_title'][:30],
                    prob['source_date']
                ])

            return rows

        load_problems_btn.click(
            fn=load_problems,
            inputs=[search_problems],
            outputs=[problems_table]
        )


# =============================================================================
# TOOLS TAB
# =============================================================================

def create_tools_tab():
    """Create the tools explorer tab."""
    with gr.Tab("Tools"):
        gr.Markdown("## Tools & Technologies")

        tools_table = gr.Dataframe(
            headers=["Tool", "Mentions"],
            label="",
            interactive=False
        )

        load_tools_btn = gr.Button("Load Tools")

        tools_chart = gr.Plot(label="Top Tools")

        def load_tools():
            tools = aggregate_tools()

            # Sort by count
            sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)
            rows = [[tool, count] for tool, count in sorted_tools]

            # Create bar chart of top 20
            top_20 = sorted_tools[:20]
            if top_20:
                fig = px.bar(
                    x=[t[0] for t in top_20],
                    y=[t[1] for t in top_20],
                    title="Top 20 Tools by Mentions",
                    labels={'x': 'Tool', 'y': 'Mentions'}
                )
                fig.update_traces(marker_color='#00ff00')
                fig.update_xaxes(tickangle=45)
                apply_matrix_theme(fig)
            else:
                fig = go.Figure()
                apply_matrix_theme(fig)

            return rows, fig

        load_tools_btn.click(
            fn=load_tools,
            outputs=[tools_table, tools_chart]
        )


# =============================================================================
# EMOTIONS TAB
# =============================================================================

def create_emotions_tab():
    """Create the emotions/tone tab."""
    with gr.Tab("Emotions"):
        gr.Markdown("## Emotional Signals")

        emotions_chart = gr.Plot(label="Tone Distribution")

        tone_filter = gr.Dropdown(
            choices=["All", "excited", "frustrated", "curious", "stuck", "neutral"],
            value="All",
            label="Filter by tone"
        )

        emotions_table = gr.Dataframe(
            headers=["Title", "Date", "Tone", "Notes"],
            label="",
            interactive=False
        )

        load_emotions_btn = gr.Button("Load Emotions")

        def load_emotions(tone_filter_val):
            emotions = aggregate_emotions()

            # Create pie chart
            tone_counts = {tone: len(convs) for tone, convs in emotions.items()}
            fig = px.pie(
                values=list(tone_counts.values()),
                names=list(tone_counts.keys()),
                title="Conversation Tones",
                color_discrete_map={
                    'excited': '#00ff00',
                    'frustrated': '#00cc00',
                    'curious': '#009900',
                    'stuck': '#006600',
                    'neutral': '#003300'
                }
            )
            apply_matrix_theme(fig)

            # Build table
            rows = []
            for tone, convs in emotions.items():
                if tone_filter_val != "All" and tone != tone_filter_val:
                    continue

                for conv in convs:
                    rows.append([
                        conv['title'][:40],
                        conv['date'],
                        tone,
                        conv['notes'][:80] if conv['notes'] else '-'
                    ])

            return fig, rows

        load_emotions_btn.click(
            fn=load_emotions,
            inputs=[tone_filter],
            outputs=[emotions_chart, emotions_table]
        )


# =============================================================================
# MAIN APP
# =============================================================================

def create_app():
    """Create the main Gradio app."""
    theme = MatrixTheme()
    with gr.Blocks(title="Resurface") as app:
        app.theme = theme
        gr.Markdown("# Resurface")

        dashboard_load_fn, dashboard_outputs = create_dashboard_tab()
        create_extraction_tab()
        create_browser_tab()
        create_ideas_tab()
        create_problems_tab()
        create_tools_tab()
        create_emotions_tab()

        # Future tabs (stubs)
        with gr.Tab("Consolidated"):
            gr.Markdown("## Consolidated View")
            gr.Markdown("*Coming soon - Phase 3*")
            gr.Markdown("This tab will show deduplicated and merged insights.")

        with gr.Tab("Categories"):
            gr.Markdown("## Categories")
            gr.Markdown("*Coming soon - Phase 4*")
            gr.Markdown("This tab will allow categorizing and scoring ideas.")

        with gr.Tab("Reports"):
            gr.Markdown("## Reports & Export")
            gr.Markdown("*Coming soon - Phase 5*")
            gr.Markdown("This tab will generate filtered views and export options.")

        # Auto-load dashboard on start
        app.load(fn=dashboard_load_fn, outputs=dashboard_outputs)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, css=MATRIX_CSS)
