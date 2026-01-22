#!/usr/bin/env python3
"""
Resurface UI - Gradio interface for conversation extraction and analysis.
"""
import json
import shutil
import subprocess
import sys
import time
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
CONSOLIDATED_DIR = Path("data/consolidated")
MANIFEST_FILE = PARSED_DIR / "_manifest.json"
CONSOLIDATED_FILE = CONSOLIDATED_DIR / "consolidated.json"
ASSETS_DIR = Path(__file__).parent / "assets"

# Set static paths for font files (must be done before creating the app)
gr.set_static_paths(paths=[str(ASSETS_DIR.absolute())])

# =============================================================================
# THEME COLOR UTILITIES
# =============================================================================

import re

def parse_color_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse color string (hex, rgb, rgba) to RGB tuple."""
    color = color.strip()

    # Handle hex format
    if color.startswith('#'):
        hex_color = color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Handle rgb/rgba format: "rgb(r, g, b)" or "rgba(r, g, b, a)"
    rgb_match = re.match(r'rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', color)
    if rgb_match:
        r, g, b = [min(255, max(0, int(float(x)))) for x in rgb_match.groups()]
        return (r, g, b)

    # Default to green if parsing fails
    return (0, 255, 0)


def generate_color_shades(color: str) -> dict:
    """Generate color shades from a base color (hex, rgb, or rgba)."""
    r, g, b = parse_color_to_rgb(color)

    def shade(factor):
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    return {
        'full': f"#{r:02x}{g:02x}{b:02x}",
        'dark': shade(0.4),      # 40% - subdued text, placeholders
        'medium': shade(0.6),    # 60% - hover states
        'dim': shade(0.2),       # 20% - row alternates, button bg
        'faint': shade(0.1),     # 10% - subtle backgrounds
    }

# =============================================================================
# MATRIX THEME
# =============================================================================

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

def create_color_palette(shades: dict):
    """Create a Gradio color palette from shades."""
    class ThemeColor(colors.Color):
        pass

    return ThemeColor(
        c50=shades['faint'],
        c100=shades['faint'],
        c200=shades['dim'],
        c300=shades['dim'],
        c400=shades['dark'],
        c500=shades['full'],
        c600=shades['full'],
        c700=shades['full'],
        c800=shades['full'],
        c900=shades['full'],
        c950=shades['full'],
    )

def create_matrix_theme(theme_color: str = "#00ff00"):
    """Create a Matrix theme with the specified color."""
    shades = generate_color_shades(theme_color)
    color_palette = create_color_palette(shades)

    class MatrixTheme(Base):
        def __init__(self):
            super().__init__(
                primary_hue=color_palette,
                secondary_hue=color_palette,
                neutral_hue=color_palette,
                font=("OpenDyslexic", "sans-serif"),
                font_mono=("OpenDyslexicMono", "monospace"),
            )
            # Override all colors dynamically
            super().set(
                # Body/background
                body_background_fill="#000000",
                body_background_fill_dark="#000000",
                body_text_color=shades['full'],
                body_text_color_dark=shades['full'],
                body_text_color_subdued=shades['dark'],
                body_text_color_subdued_dark=shades['dark'],

                # Blocks
                block_background_fill="#000000",
                block_background_fill_dark="#000000",
                block_border_color=shades['full'],
                block_border_color_dark=shades['full'],
                block_label_background_fill="#000000",
                block_label_background_fill_dark="#000000",
                block_label_text_color=shades['full'],
                block_label_text_color_dark=shades['full'],
                block_title_text_color=shades['full'],
                block_title_text_color_dark=shades['full'],

                # Panels
                panel_background_fill="#000000",
                panel_background_fill_dark="#000000",
                panel_border_color=shades['full'],
                panel_border_color_dark=shades['full'],

                # Buttons
                button_primary_background_fill=shades['dim'],
                button_primary_background_fill_dark=shades['dim'],
                button_primary_background_fill_hover=shades['dark'],
                button_primary_background_fill_hover_dark=shades['dark'],
                button_primary_text_color=shades['full'],
                button_primary_text_color_dark=shades['full'],
                button_primary_border_color=shades['full'],
                button_primary_border_color_dark=shades['full'],
                button_secondary_background_fill="#000000",
                button_secondary_background_fill_dark="#000000",
                button_secondary_background_fill_hover=shades['faint'],
                button_secondary_background_fill_hover_dark=shades['faint'],
                button_secondary_text_color=shades['full'],
                button_secondary_text_color_dark=shades['full'],
                button_secondary_border_color=shades['full'],
                button_secondary_border_color_dark=shades['full'],
                button_cancel_background_fill="#000000",
                button_cancel_background_fill_dark="#000000",
                button_cancel_text_color=shades['full'],
                button_cancel_text_color_dark=shades['full'],

                # Inputs
                input_background_fill="#000000",
                input_background_fill_dark="#000000",
                input_border_color=shades['full'],
                input_border_color_dark=shades['full'],
                input_border_color_focus=shades['full'],
                input_border_color_focus_dark=shades['full'],
                input_placeholder_color=shades['dark'],
                input_placeholder_color_dark=shades['dark'],

                # Tables
                table_border_color=shades['full'],
                table_border_color_dark=shades['full'],
                table_even_background_fill="#000000",
                table_even_background_fill_dark="#000000",
                table_odd_background_fill=shades['faint'],
                table_odd_background_fill_dark=shades['faint'],
                table_row_focus=shades['dim'],
                table_row_focus_dark=shades['dim'],

                # Checkboxes
                checkbox_background_color="#000000",
                checkbox_background_color_dark="#000000",
                checkbox_background_color_selected=shades['full'],
                checkbox_background_color_selected_dark=shades['full'],
                checkbox_border_color=shades['full'],
                checkbox_border_color_dark=shades['full'],
                checkbox_label_text_color=shades['full'],
                checkbox_label_text_color_dark=shades['full'],

                # Sliders
                slider_color=shades['full'],
                slider_color_dark=shades['full'],

                # Links
                link_text_color=shades['full'],
                link_text_color_dark=shades['full'],
                link_text_color_hover=shades['medium'],
                link_text_color_hover_dark=shades['medium'],
                link_text_color_visited=shades['medium'],
                link_text_color_visited_dark=shades['medium'],
                link_text_color_active=shades['full'],
                link_text_color_active_dark=shades['full'],

                # Borders
                border_color_accent=shades['full'],
                border_color_accent_dark=shades['full'],
                border_color_primary=shades['full'],
                border_color_primary_dark=shades['full'],

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
                color_accent=shades['full'],
                color_accent_soft=shades['dim'],
                color_accent_soft_dark=shades['dim'],
            )

    return MatrixTheme()

# Additional CSS for elements theme doesn't cover
def generate_matrix_css(theme_color: str = "#00ff00") -> str:
    """Generate CSS with the specified theme color using CSS variables."""
    shades = generate_color_shades(theme_color)

    return f"""
/* CSS Custom Properties for Theme Color */
:root {{
    --theme-color: {shades['full']};
    --theme-color-dark: {shades['dark']};
    --theme-color-medium: {shades['medium']};
    --theme-color-dim: {shades['dim']};
    --theme-color-faint: {shades['faint']};
    --bg-color: #000000;
    color-scheme: dark !important;
}}

/* OpenDyslexic Font - Self-Hosted for Offline Use */
@font-face {{
    font-family: 'OpenDyslexic';
    src: url('/gradio_api/file=assets/fonts/OpenDyslexic-Regular.woff') format('woff');
    font-weight: normal;
    font-style: normal;
    font-display: swap;
}}

@font-face {{
    font-family: 'OpenDyslexic';
    src: url('/gradio_api/file=assets/fonts/OpenDyslexic-Bold.woff') format('woff');
    font-weight: bold;
    font-style: normal;
    font-display: swap;
}}

@font-face {{
    font-family: 'OpenDyslexicMono';
    src: url('/gradio_api/file=assets/fonts/OpenDyslexicMono-Regular.otf') format('opentype');
    font-weight: normal;
    font-style: normal;
    font-display: swap;
}}

/* Apply dyslexia-friendly font globally */
* {{
    font-family: 'OpenDyslexic', sans-serif !important;
}}

code, pre, .mono, .gr-code, textarea {{
    font-family: 'OpenDyslexicMono', monospace !important;
}}

/* Remove Gradio footer */
footer {{
    display: none !important;
}}

/* Scrollbars */
::-webkit-scrollbar {{
    background: var(--bg-color);
    width: 8px;
}}
::-webkit-scrollbar-thumb {{
    background: var(--theme-color);
    border-radius: 4px;
}}
::-webkit-scrollbar-track {{
    background: var(--theme-color-faint);
}}

/* Tab styling */
.tab-nav button {{
    background: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border: 1px solid var(--theme-color) !important;
}}
.tab-nav button.selected {{
    background: var(--theme-color-dim) !important;
}}

/* Tab underline - remove orange */
.tab-nav button.selected::after,
button.tab-nav-button.selected::after,
.tab-nav button::after,
button.tab-nav-button::after {{
    background: var(--theme-color) !important;
    background-color: var(--theme-color) !important;
    border-bottom-color: var(--theme-color) !important;
}}
.tabs .tabitem {{
    border-color: var(--theme-color) !important;
}}
/* More aggressive tab underline override */
[role="tablist"] button[aria-selected="true"]::after,
[role="tablist"] button.selected::after,
.tab-nav .selected::after,
.svelte-1kcgrqr::after,
button[role="tab"][aria-selected="true"]::after {{
    background: var(--theme-color) !important;
    background-color: var(--theme-color) !important;
}}
/* Target any orange color directly */
[style*="rgb(249, 115, 22)"],
[style*="#f97316"],
[style*="orange"] {{
    background: var(--theme-color) !important;
    background-color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Primary buttons - force theme styling */
button.primary,
button.lg.primary,
button[variant="primary"],
.primary-btn,
.gr-button-primary {{
    background: var(--theme-color-dim) !important;
    background-color: var(--theme-color-dim) !important;
    color: var(--theme-color) !important;
    border: 1px solid var(--theme-color) !important;
}}
button.primary:hover,
button.lg.primary:hover {{
    background: var(--theme-color-dark) !important;
    background-color: var(--theme-color-dark) !important;
}}

/* Secondary buttons - outlined style with border */
button.secondary,
button[variant="secondary"],
.secondary-btn,
.gr-button-secondary {{
    background: transparent !important;
    background-color: transparent !important;
    color: var(--theme-color) !important;
    border: 1px solid var(--theme-color) !important;
}}
button.secondary:hover,
button[variant="secondary"]:hover {{
    background: var(--theme-color-faint) !important;
    background-color: var(--theme-color-faint) !important;
}}

/* Slider track */
input[type="range"] {{
    accent-color: var(--theme-color) !important;
}}
input[type="range"]::-webkit-slider-runnable-track {{
    background: linear-gradient(to right, var(--theme-color) 0%, var(--theme-color) var(--value-percent, 50%), var(--theme-color-dim) var(--value-percent, 50%), var(--theme-color-dim) 100%) !important;
}}
input[type="range"]::-moz-range-track {{
    background: var(--theme-color-dim) !important;
}}
input[type="range"]::-moz-range-progress {{
    background: var(--theme-color) !important;
}}
.range-slider,
.slider {{
    --slider-color: var(--theme-color) !important;
}}

/* Radio buttons */
input[type="radio"] {{
    accent-color: var(--theme-color) !important;
}}
input[type="radio"]:checked {{
    background-color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}
.gr-radio-row input[type="radio"]:checked + label::before {{
    background-color: var(--theme-color) !important;
}}

/* Ensure all text uses theme color */
* {{
    color: var(--theme-color);
}}

/* SVG icons */
svg path, svg circle, svg rect {{
    stroke: var(--theme-color) !important;
}}

/* Force all gray backgrounds to pure black */
.block, .form, .panel, .container, .wrap, .wrap-inner,
.gr-box, .gr-panel, .gr-form, .gr-block, .gr-padded,
.gradio-container, .contain, .gap, .gr-group,
[class*="block"], [class*="panel"], [class*="container"],
.svelte-1ed2p3z, .svelte-1kcgrqr, .svelte-1guhx2a,
div[data-testid], main, section, article, aside,
.input-container, .output-container, .component-wrapper {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Table backgrounds */
table, thead, tbody, tr, th, td,
.table-wrap, .dataframe, .gr-dataframe {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}
table tr:nth-child(even),
table tr:nth-child(odd) {{
    background: var(--bg-color) !important;
}}
table tr:hover {{
    background: var(--theme-color-faint) !important;
}}

/* Input/textarea backgrounds */
input, textarea, select, .gr-input, .gr-textarea,
.textbox, .gr-textbox, [data-testid="textbox"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Dropdown/select backgrounds */
.dropdown, .gr-dropdown, select, option,
[data-testid="dropdown"], .choices, .choices__inner {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

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
[style*="#6b7280"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Body and html */
html, body {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Additional targeted selectors for remaining gray elements */
.gr-box, .gr-panel, .gr-form, .gr-input, .gr-button,
.secondary, .gr-secondary, .svelte-1ipelgc, .svelte-1ed2p3z,
.border, .rounded, .shadow, .bg-gray-50, .bg-gray-100,
.bg-gray-200, .bg-gray-700, .bg-gray-800, .bg-gray-900,
div[class*="svelte-"], span[class*="svelte-"] {{
    background-color: var(--bg-color) !important;
}}

/* Borders should use theme color */
.gr-box, .gr-panel, .gr-form, .gr-input, .gr-button,
.border, input, textarea, select, button, table, th, td {{
    border-color: var(--theme-color) !important;
}}

/* Remove any box shadows that might appear gray */
.gr-box, .gr-panel, .shadow, [class*="shadow"] {{
    box-shadow: none !important;
}}

/* JSON viewer styling */
.json-holder, .json-container, [data-testid="json"],
.gr-json, .json, pre, code,
[class*="json"], [class*="Json"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}
.json-holder pre, .json-container pre,
[data-testid="json"] pre {{
    background: var(--bg-color) !important;
    color: var(--theme-color) !important;
}}
/* JSON syntax highlighting */
.json-holder .string, .json-container .string {{ color: var(--theme-color-medium) !important; }}
.json-holder .number, .json-container .number {{ color: var(--theme-color) !important; }}
.json-holder .boolean, .json-container .boolean {{ color: var(--theme-color) !important; }}
.json-holder .null, .json-container .null {{ color: var(--theme-color-dark) !important; }}
.json-holder .key, .json-container .key {{ color: var(--theme-color) !important; }}

/* Checkbox styling - more specific */
input[type="checkbox"] {{
    accent-color: var(--theme-color) !important;
    background: var(--bg-color) !important;
    border-color: var(--theme-color) !important;
}}
input[type="checkbox"]:checked {{
    background-color: var(--theme-color) !important;
}}
.gr-checkbox, .checkbox-container, [data-testid="checkbox"],
label[data-testid], .gr-check-radio {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}
/* Checkbox label and wrapper */
.gr-checkbox-container, .checkbox-label,
[class*="checkbox"], [class*="Checkbox"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
}}

/* Dropdown menu items */
.dropdown-menu, .dropdown-content, .dropdown-item,
.choices__list, .choices__item, .choices__list--dropdown,
ul[role="listbox"], li[role="option"],
[class*="dropdown"], [class*="Dropdown"],
.svelte-select, .listbox, .list-box {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}
.choices__list--dropdown .choices__item--selectable.is-highlighted,
li[role="option"]:hover, .dropdown-item:hover {{
    background: var(--theme-color-dim) !important;
    background-color: var(--theme-color-dim) !important;
}}

/* Label backgrounds */
label, .label, .gr-label, [class*="label"],
.block-label, .label-wrap, span.label {{
    background: transparent !important;
    background-color: transparent !important;
    color: var(--theme-color) !important;
}}

/* Number input styling */
input[type="number"], .gr-number,
[data-testid="number-input"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Slider label and container */
.slider-container, .range-container,
[data-testid="slider"], .gr-slider {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Password input */
input[type="password"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Prose/markdown containers */
.prose, .markdown, .md, [class*="prose"],
[class*="markdown"], [class*="Markdown"] {{
    background: transparent !important;
    background-color: transparent !important;
    color: var(--theme-color) !important;
}}
.prose *, .markdown *, .md * {{
    color: var(--theme-color) !important;
}}

/* Catch-all for white backgrounds */
[style*="background: white"],
[style*="background-color: white"],
[style*="background: #fff"],
[style*="background-color: #fff"],
[style*="background: rgb(255"],
[style*="background-color: rgb(255"],
[style*="background: #ffffff"],
[style*="background-color: #ffffff"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Nested tabs in Consolidated view */
.tabs .tabs, .tabitem .tabs,
.tab-content, .tabitem-content {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
}}

/* Form elements wrapper */
.form-group, .input-group, .field-group,
.gr-form-group, fieldset, legend {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Accordion/collapsible elements */
.accordion, .collapsible, details, summary,
[class*="accordion"], [class*="Accordion"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Status/info boxes */
.status, .info, .alert, .notice, .message,
[class*="status"], [class*="info"], [class*="alert"] {{
    background: var(--bg-color) !important;
    background-color: var(--bg-color) !important;
    color: var(--theme-color) !important;
    border-color: var(--theme-color) !important;
}}

/* Ensure dividers use theme color */
hr, .divider, [class*="divider"] {{
    border-color: var(--theme-color) !important;
    background: var(--theme-color) !important;
}}

/* Dataframe and component labels - very specific */
.gr-dataframe label, .dataframe label,
[data-testid="dataframe"] label,
.gr-dataframe span, .dataframe span,
[data-testid="dataframe"] span,
.table-wrap label, .table-wrap span,
.gr-dataframe .label, .dataframe .label,
.block .label, .block label, .block span.svelte-1gfkn6j,
span.svelte-1gfkn6j, label.svelte-1gfkn6j,
.svelte-1gfkn6j, .svelte-1f354aw, .svelte-s1r2yt,
span[data-testid], label[data-testid],
.label-wrap span, .label-wrap label,
.gr-block-label, .gr-box-label,
h2, h3, h4, h5, h6 {{
    color: var(--theme-color) !important;
    background: transparent !important;
    background-color: transparent !important;
}}

/* Target all span and label elements more aggressively */
span, label {{
    color: var(--theme-color) !important;
}}

/* Nuclear option - catch ANY white text */
[style*="color: white"],
[style*="color: #fff"],
[style*="color: rgb(255"],
[style*="color:#fff"],
[style*="color:#ffffff"],
[style*="color: #ffffff"] {{
    color: var(--theme-color) !important;
}}

/* Gradio 4.x specific label selectors */
.block > label, .block > span,
.wrap > label, .wrap > span,
.container > label, .container > span,
div > label, div > span,
.gr-block > label, .gr-block > span,
.gr-group > label, .gr-group > span,
[class*="block"] > label, [class*="block"] > span,
[class*="wrap"] > label, [class*="wrap"] > span {{
    color: var(--theme-color) !important;
}}

/* Even more specific - target by structure */
.gradio-container label,
.gradio-container span,
.gradio-container p,
.gradio-container div {{
    color: var(--theme-color) !important;
}}

/* Override any text color */
[class*="svelte-"] {{
    color: var(--theme-color) !important;
}}

/* Plotly chart labels if any remain */
.plotly .gtitle, .plotly .xtitle, .plotly .ytitle,
.plotly text, .js-plotly-plot text {{
    fill: var(--theme-color) !important;
    color: var(--theme-color) !important;
}}

/* ==========================================================================
   COMPACT VERTICAL LAYOUT - Maximum Space Efficiency
   ========================================================================== */

/* Global line height reduction */
* {{
    line-height: 1.3 !important;
}}

/* Main container - minimal padding */
.gradio-container {{
    padding: 4px 8px !important;
}}

/* Blocks and panels - tight vertical spacing */
.gr-block, .block, .gr-box, .gr-panel, .gr-form,
.gr-group, .group, .wrap, .contain {{
    padding: 4px 6px !important;
    margin: 2px 0 !important;
}}

/* Row and column gaps - minimal */
.gr-row, .row, .gap, .flex {{
    gap: 6px !important;
    margin: 2px 0 !important;
}}
.gr-column, .column {{
    gap: 4px !important;
}}

/* Headings - compact with clear hierarchy */
h1, .prose h1 {{
    margin: 4px 0 2px 0 !important;
    padding: 0 !important;
    font-size: 1.4em !important;
}}
h2, .prose h2 {{
    margin: 6px 0 2px 0 !important;
    padding: 0 !important;
    font-size: 1.2em !important;
}}
h3, .prose h3 {{
    margin: 4px 0 2px 0 !important;
    padding: 0 !important;
    font-size: 1.1em !important;
}}
h4, h5, h6, .prose h4, .prose h5, .prose h6 {{
    margin: 2px 0 !important;
    padding: 0 !important;
}}

/* Paragraphs and text blocks */
p, .prose p, .markdown p {{
    margin: 2px 0 !important;
    padding: 0 !important;
}}

/* Tab navigation - compact */
.tab-nav {{
    margin-bottom: 4px !important;
    gap: 2px !important;
}}

/* Tab content area */
.tabitem, .tab-content, .tabitem > div {{
    padding: 6px !important;
    margin: 0 !important;
}}

/* Form elements - tight */
textarea {{
    min-height: unset !important;
}}

/* Number inputs */
input[type="number"] {{
    padding: 4px 6px !important;
}}

/* Buttons - compact */
button.lg {{
    padding: 6px 12px !important;
}}

/* Dataframes and tables - compact */
.dataframe, table, .gr-dataframe {{
    margin: 4px 0 !important;
}}
.dataframe td, .dataframe th, td, th {{
    padding: 3px 6px !important;
    line-height: 1.2 !important;
}}
.dataframe thead, thead {{
    line-height: 1.2 !important;
}}

/* Checkbox and radio groups */
.gr-checkbox, .gr-radio, .checkbox-group, .radio-group {{
    margin: 2px 0 !important;
    padding: 2px !important;
}}
.gr-checkbox-container, .checkbox-container {{
    gap: 4px !important;
}}

/* Dropdown menus */
.dropdown, .gr-dropdown {{
    margin: 2px 0 !important;
}}

/* Accordion/collapsible - tight */
.accordion, details, summary {{
    margin: 2px 0 !important;
    padding: 4px !important;
}}

/* Status and info boxes */
.status, .info, .alert, .message {{
    padding: 4px 8px !important;
    margin: 2px 0 !important;
}}

/* Dividers - thin */
hr, .divider {{
    margin: 6px 0 !important;
}}

/* JSON displays */
.json-holder, .json-container, pre {{
    padding: 4px !important;
    margin: 2px 0 !important;
    line-height: 1.2 !important;
}}

/* Plotly charts - reduce wrapper padding */
.js-plotly-plot, .plotly-graph-div {{
    margin: 4px 0 !important;
}}

/* Remove extra spacing from nested containers */
.block > .wrap, .block > .container,
.gr-block > .wrap, .gr-block > .container {{
    padding: 2px !important;
    margin: 0 !important;
}}

/* Svelte component wrappers - reduce gaps */
[class*="svelte-"] {{
    --block-padding: 4px !important;
    --block-gap: 4px !important;
    --layout-gap: 4px !important;
}}
"""

# =============================================================================
# PLOTLY MATRIX THEME HELPER
# =============================================================================

def apply_matrix_theme(fig, theme_color: str = None):
    """Apply matrix theme to a Plotly figure."""
    from config import load_config
    if theme_color is None:
        config = load_config()
        theme_color = config.get('theme_color', '#00ff00')

    shades = generate_color_shades(theme_color)

    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color=shades['full'], family='OpenDyslexic, Courier New'),
        title_font=dict(color=shades['full']),
        legend=dict(font=dict(color=shades['full'])),
        xaxis=dict(
            gridcolor=shades['dim'],
            linecolor=shades['full'],
            tickfont=dict(color=shades['full'])
        ),
        yaxis=dict(
            gridcolor=shades['dim'],
            linecolor=shades['full'],
            tickfont=dict(color=shades['full'])
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
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
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
            with open(extraction_file, 'r', encoding='utf-8') as f:
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
        with open(f, 'r', encoding='utf-8') as file:
            extractions.append(json.load(file))

    return extractions


def load_conversation(conv_id: str) -> dict | None:
    """Load a parsed conversation by ID."""
    conv_file = PARSED_DIR / f"{conv_id}.json"
    if not conv_file.exists():
        return None
    with open(conv_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_extraction(conv_id: str) -> dict | None:
    """Load extraction for a conversation."""
    extraction_file = EXTRACTIONS_DIR / f"{conv_id}.json"
    if not extraction_file.exists():
        return None
    with open(extraction_file, 'r', encoding='utf-8') as f:
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
    with gr.Tab("Dashboard", id="dashboard") as tab:
        with gr.Row():
            total_box = gr.Number(label="Total Conversations", interactive=False)
            extracted_box = gr.Number(label="Extracted", interactive=False)
            remaining_box = gr.Number(label="Remaining", interactive=False)
            errors_box = gr.Number(label="Errors", interactive=False)

        gr.Markdown("### Extraction Progress")
        progress_bar = gr.Markdown("0% complete")

        gr.Markdown("### Quick Stats")
        with gr.Row():
            ideas_count = gr.Number(label="Total Ideas", interactive=False)
            problems_count = gr.Number(label="Total Problems", interactive=False)
            tools_count = gr.Number(label="Unique Tools", interactive=False)

        refresh_btn = gr.Button("Refresh Dashboard", variant="primary")

        def refresh_dashboard():
            status = get_status()
            extractions = load_all_extractions()

            # Quick stats
            ideas = aggregate_ideas()
            problems = aggregate_problems()
            tools = aggregate_tools()

            # Create visual progress bar
            pct = status['progress']
            filled = int(pct / 2)  # 50 chars total
            empty = 50 - filled
            progress_text = f"**{pct:.1f}%** `[{'█' * filled}{'░' * empty}]` ({status['extracted']:,} / {status['total']:,})"

            return (
                status['total'],
                status['extracted'],
                status['remaining'],
                status['errors'],
                progress_text,
                len(ideas),
                len(problems),
                len(tools)
            )

        refresh_btn.click(
            fn=refresh_dashboard,
            outputs=[
                total_box, extracted_box, remaining_box, errors_box,
                progress_bar, ideas_count, problems_count, tools_count
            ]
        )

    return (tab, refresh_dashboard, [
        total_box, extracted_box, remaining_box, errors_box,
        progress_bar, ideas_count, problems_count, tools_count
    ])


# =============================================================================
# EXTRACTION CONTROL TAB
# =============================================================================

def format_time(seconds: float | None) -> str:
    """Format seconds as human-readable time."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def create_extraction_tab():
    """Create the extraction control tab."""
    from config import load_config

    def get_settings_display():
        """Get current settings as formatted markdown."""
        config = load_config()
        provider = config.get('api_provider', 'openai')
        if provider == 'ollama':
            model = config.get('ollama_model', 'unknown')
        else:
            model = config.get('model', 'unknown')
        return f"""**Model:** {model}
**Provider:** {provider}
**Rate limit:** {config.get('requests_per_minute', 60)} req/min"""

    with gr.Tab("Extraction"):
        gr.Markdown("## Extraction Control")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current Settings")
                settings_display = gr.Markdown(get_settings_display())
                refresh_settings_btn = gr.Button("↻ Refresh", size="sm", variant="secondary")

            with gr.Column():
                gr.Markdown("### Run Extraction")
                count_input = gr.Number(
                    label="Number to extract",
                    value=10,
                    minimum=1,
                    step=1
                )
                auto_full_process = gr.Checkbox(
                    label="Auto-run full process after extraction",
                    value=False,
                    info="Runs consolidation, categorization, and synthesis automatically"
                )
                with gr.Row():
                    extract_btn = gr.Button("Start Extraction", variant="primary")
                    extract_all_btn = gr.Button("Extract All Remaining", variant="secondary")

        # Refresh settings display when button clicked
        refresh_settings_btn.click(
            fn=get_settings_display,
            outputs=[settings_display]
        )

        output_log = gr.Textbox(
            label="Extraction Log",
            lines=15,
            max_lines=30,
            interactive=False
        )

        gr.Markdown("### Extract Specific Conversation")
        with gr.Row():
            conv_id_input = gr.Textbox(label="Conversation ID", placeholder="Enter UUID...")
            extract_one_btn = gr.Button("Extract This", variant="secondary")

        # Full Process Pipeline section
        gr.Markdown("---")
        gr.Markdown("### Automated Processing Pipeline")
        gr.Markdown("After extracting conversations, run the full analysis pipeline: **Consolidation → Categorization → Synthesis**")

        with gr.Row():
            run_full_process_btn = gr.Button(
                "Run Full Process on Extracted Conversations",
                variant="primary",
                size="lg"
            )

        full_process_log = gr.Textbox(
            label="Pipeline Log",
            value="Click 'Run Full Process' to execute all analysis steps sequentially on your extracted conversations.",
            lines=20,
            max_lines=40,
            interactive=False
        )

        def run_extraction_with_polling(count=None, extract_all=False):
            """Run extraction with polling for status updates."""
            import time as time_module

            STATUS_FILE = Path("data/extraction_status.json")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if extract_all:
                yield f"[{timestamp}] Starting extraction of ALL remaining conversations...\n\nThis may take a while. Please wait..."
                cmd = [sys.executable, "runner.py", "--all"]
            else:
                yield f"[{timestamp}] Starting extraction of {int(count)} conversations...\n\nPlease wait..."
                cmd = [sys.executable, "runner.py", "--count", str(int(count))]

            # Clear old status file
            if STATUS_FILE.exists():
                STATUS_FILE.unlink()

            try:
                # Start process in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                    text=True
                )

                start_time = time_module.time()
                last_message = ""

                # Poll for status updates
                while process.poll() is None:  # Process still running
                    if STATUS_FILE.exists():
                        try:
                            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                                status = json.load(f)
                            message = status.get('message', '')
                            progress = status.get('progress', 0)
                            current = status.get('current', 0)
                            total = status.get('total', 0)
                            elapsed = status.get('elapsed_seconds', 0)
                            eta = status.get('eta_seconds')

                            if message != last_message or True:  # Always update for time
                                last_message = message
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                elapsed_str = format_time(elapsed)
                                eta_str = format_time(eta) if eta else "calculating..."

                                progress_bar = f"[{'█' * int(progress/5)}{'░' * (20-int(progress/5))}] {progress:.0f}%"
                                yield f"[{ts}] [{current}/{total}] {message}\nElapsed: {elapsed_str} | ETA: {eta_str}\n{progress_bar}"

                            if status.get('complete') or status.get('error'):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass  # Status file being written

                    # Timeout after 2 hours for large extractions
                    if time_module.time() - start_time > 7200:
                        process.kill()
                        yield f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Process timed out after 2 hours."
                        return

                    time_module.sleep(1)  # Poll every 1 second

                # Get final output
                stdout, stderr = process.communicate(timeout=10)
                end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Check final status
                if STATUS_FILE.exists():
                    try:
                        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        elapsed = status.get('elapsed_seconds', 0)
                        elapsed_str = format_time(elapsed)

                        if status.get('error'):
                            yield f"[{end_timestamp}] {status.get('message', 'Unknown error')}"
                            return
                        if status.get('complete'):
                            yield f"[{end_timestamp}] {status.get('message', 'Extraction complete.')}\nTotal time: {elapsed_str}"
                            return
                    except (json.JSONDecodeError, IOError):
                        pass

                if process.returncode == 0:
                    yield f"[{end_timestamp}] Extraction complete."
                else:
                    yield f"[{end_timestamp}] Extraction finished with errors.\n\n{stderr}"

            except Exception as e:
                yield f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}"

        def run_extraction(count):
            yield from run_extraction_with_polling(count=count, extract_all=False)

        def run_all_extraction():
            yield from run_extraction_with_polling(extract_all=True)

        def run_extraction_with_auto_process(count, auto_process):
            """Run extraction and optionally chain into full process."""
            STATUS_FILE = Path("data/extraction_status.json")

            # Run extraction
            for msg in run_extraction_with_polling(count=count, extract_all=False):
                yield msg

            # Check if we should auto-continue
            if not auto_process:
                return

            # Check if extraction succeeded
            extraction_succeeded = False
            if STATUS_FILE.exists():
                try:
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                    extraction_succeeded = status.get('complete', False) and not status.get('error', False)
                except (json.JSONDecodeError, IOError):
                    pass

            if extraction_succeeded:
                yield "\n\n" + "=" * 60 + "\nAUTO-STARTING FULL PROCESS...\n" + "=" * 60 + "\n"
                for msg in run_full_process():
                    yield msg
            else:
                yield "\n\nSkipping auto-run: extraction did not complete successfully."

        def run_all_extraction_with_auto_process(auto_process):
            """Run extract all and optionally chain into full process."""
            STATUS_FILE = Path("data/extraction_status.json")

            # Run extraction
            for msg in run_extraction_with_polling(extract_all=True):
                yield msg

            # Check if we should auto-continue
            if not auto_process:
                return

            # Check if extraction succeeded
            extraction_succeeded = False
            if STATUS_FILE.exists():
                try:
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                    extraction_succeeded = status.get('complete', False) and not status.get('error', False)
                except (json.JSONDecodeError, IOError):
                    pass

            if extraction_succeeded:
                yield "\n\n" + "=" * 60 + "\nAUTO-STARTING FULL PROCESS...\n" + "=" * 60 + "\n"
                for msg in run_full_process():
                    yield msg
            else:
                yield "\n\nSkipping auto-run: extraction did not complete successfully."

        def run_single_extraction(conv_id):
            if not conv_id.strip():
                return "Please enter a conversation ID"
            try:
                result = subprocess.run(
                    [sys.executable, "runner.py", "--id", conv_id.strip()],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                return result.stdout + result.stderr
            except Exception as e:
                return f"Error: {e}"

        def run_full_process():
            """Execute full analysis pipeline: Consolidation → Categorization → Synthesis."""
            from synthesizer import run_synthesis

            start_time = time.time()
            overall_log = []

            # Check prerequisites
            extraction_files = list(EXTRACTIONS_DIR.glob("*.json"))
            extraction_count = len([f for f in extraction_files if not f.name.startswith("_")])

            if extraction_count == 0:
                yield "Error: No extracted conversations found.\n\nPlease run extraction first before running the full process."
                return

            overall_log.append("=" * 60)
            overall_log.append("FULL PROCESS PIPELINE")
            overall_log.append("=" * 60)
            overall_log.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            overall_log.append(f"Extracted conversations: {extraction_count}")
            overall_log.append("")
            yield "\n".join(overall_log)

            # ================================================================
            # PHASE 1: CONSOLIDATION
            # ================================================================
            overall_log.append("-" * 60)
            overall_log.append("PHASE 1: CONSOLIDATION")
            overall_log.append("-" * 60)
            yield "\n".join(overall_log)

            consolidation_status_file = Path("data/consolidation_status.json")
            if consolidation_status_file.exists():
                consolidation_status_file.unlink()

            try:
                consolidation_process = subprocess.Popen(
                    [sys.executable, "consolidate.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                    text=True
                )

                phase_start = time.time()
                last_message = ""

                while consolidation_process.poll() is None:
                    if consolidation_status_file.exists():
                        try:
                            with open(consolidation_status_file, 'r', encoding='utf-8') as f:
                                status = json.load(f)

                            message = status.get('message', '')
                            progress = status.get('progress', 0) or 0

                            if message != last_message:
                                last_message = message
                                progress_bar = f"[{'█' * int(progress/5)}{'░' * (20-int(progress/5))}] {progress:.0f}%"
                                overall_log.append(f"  {message} {progress_bar}")
                                yield "\n".join(overall_log)

                            if status.get('complete') or status.get('error'):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass

                    time.sleep(2)

                stdout, stderr = consolidation_process.communicate(timeout=10)

                if consolidation_process.returncode == 0:
                    phase_time = time.time() - phase_start
                    overall_log.append(f"  ✓ Consolidation complete ({format_time(phase_time)})")
                    overall_log.append("")
                    yield "\n".join(overall_log)
                else:
                    overall_log.append(f"  ✗ Consolidation failed")
                    if stderr:
                        overall_log.append(f"  Error: {stderr[:500]}")
                    yield "\n".join(overall_log)
                    return

            except Exception as e:
                overall_log.append(f"  ✗ Error: {e}")
                yield "\n".join(overall_log)
                return

            # ================================================================
            # PHASE 2: CATEGORIZATION
            # ================================================================
            overall_log.append("-" * 60)
            overall_log.append("PHASE 2: CATEGORIZATION")
            overall_log.append("-" * 60)
            yield "\n".join(overall_log)

            categorization_status_file = Path("data/categorization_status.json")
            if categorization_status_file.exists():
                categorization_status_file.unlink()

            try:
                categorization_process = subprocess.Popen(
                    [sys.executable, "categorize.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                    text=True
                )

                phase_start = time.time()
                last_message = ""

                while categorization_process.poll() is None:
                    if categorization_status_file.exists():
                        try:
                            with open(categorization_status_file, 'r', encoding='utf-8') as f:
                                status = json.load(f)

                            message = status.get('message', '')
                            progress = status.get('progress', 0) or 0

                            if message != last_message:
                                last_message = message
                                progress_bar = f"[{'█' * int(progress/5)}{'░' * (20-int(progress/5))}] {progress:.0f}%"
                                overall_log.append(f"  {message} {progress_bar}")
                                yield "\n".join(overall_log)

                            if status.get('complete') or status.get('error'):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass

                    time.sleep(2)

                stdout, stderr = categorization_process.communicate(timeout=10)

                if categorization_process.returncode == 0:
                    phase_time = time.time() - phase_start
                    overall_log.append(f"  ✓ Categorization complete ({format_time(phase_time)})")
                    overall_log.append("")
                    yield "\n".join(overall_log)
                else:
                    overall_log.append(f"  ✗ Categorization failed")
                    if stderr:
                        overall_log.append(f"  Error: {stderr[:500]}")
                    yield "\n".join(overall_log)
                    return

            except Exception as e:
                overall_log.append(f"  ✗ Error: {e}")
                yield "\n".join(overall_log)
                return

            # ================================================================
            # PHASE 3: SYNTHESIS
            # ================================================================
            overall_log.append("-" * 60)
            overall_log.append("PHASE 3: SYNTHESIS")
            overall_log.append("-" * 60)
            yield "\n".join(overall_log)

            try:
                phase_start = time.time()
                overall_log.append("  Starting synthesis generation...")
                yield "\n".join(overall_log)

                result = run_synthesis()

                if result:
                    phase_time = time.time() - phase_start
                    overall_log.append(f"  ✓ Synthesis complete ({format_time(phase_time)})")

                    # Add summary stats
                    profile = result.get("profile", {})
                    all_ideas = result.get("all_ideas", [])
                    overall_log.append(f"  Generated {len(all_ideas)} project ideas")
                    if profile:
                        core_themes = profile.get('core_themes', [])
                        overall_log.append(f"  Core themes identified: {len(core_themes)}")
                    overall_log.append("")
                    yield "\n".join(overall_log)
                else:
                    overall_log.append("  ✗ Synthesis returned no results")
                    yield "\n".join(overall_log)
                    return

            except Exception as e:
                overall_log.append(f"  ✗ Error: {e}")
                yield "\n".join(overall_log)
                return

            # ================================================================
            # COMPLETION
            # ================================================================
            total_time = time.time() - start_time
            overall_log.append("=" * 60)
            overall_log.append("✓ PIPELINE COMPLETE")
            overall_log.append("=" * 60)
            overall_log.append(f"Total time: {format_time(total_time)}")
            overall_log.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            overall_log.append("")
            overall_log.append("Navigate to the Consolidated, Categories, and Synthesis tabs to view results.")

            yield "\n".join(overall_log)

        extract_btn.click(fn=run_extraction_with_auto_process, inputs=[count_input, auto_full_process], outputs=[output_log])
        extract_all_btn.click(fn=run_all_extraction_with_auto_process, inputs=[auto_full_process], outputs=[output_log])
        extract_one_btn.click(fn=run_single_extraction, inputs=[conv_id_input], outputs=[output_log])
        run_full_process_btn.click(fn=run_full_process, inputs=[], outputs=[full_process_log])


# =============================================================================
# CONVERSATION BROWSER TAB
# =============================================================================

def create_browser_tab():
    """Create the conversation browser tab."""
    PAGE_SIZE = 100

    with gr.Tab("Conversations") as tab:
        gr.Markdown("## Conversation Browser")

        with gr.Row():
            search_box = gr.Textbox(label="Search titles", placeholder="Type to search...")
            filter_extracted = gr.Checkbox(label="Only show extracted", value=False)
            sort_dropdown = gr.Dropdown(
                choices=["Oldest first", "Newest first", "Most turns", "Fewest turns"],
                value="Oldest first",
                label="Sort by"
            )

        with gr.Row():
            load_btn = gr.Button("Load Conversations", variant="primary")
            prev_btn = gr.Button("< Previous", variant="secondary")
            page_info = gr.Markdown("Page 1 of 1 (0 conversations)")
            next_btn = gr.Button("Next >", variant="secondary")

        conversations_table = gr.Dataframe(
            headers=["Title", "Date", "Messages", "Extracted", "ID"],
            label="Conversations",
            interactive=False
        )

        # State for pagination
        current_page = gr.State(1)
        total_pages = gr.State(1)
        filtered_data = gr.State([])

        gr.Markdown("### Conversation Detail")
        conv_id_display = gr.Textbox(label="Selected ID", interactive=False)

        with gr.Tabs():
            with gr.Tab("Messages"):
                messages_display = gr.Markdown("Select a conversation to view messages")

            with gr.Tab("Extraction"):
                extraction_display = gr.JSON(label="Extraction Data")

        def load_conversations(search_term, only_extracted, sort_by):
            """Load and filter all conversations, return first page."""
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

            # Sort based on selection
            if sort_by == "Oldest first":
                rows.sort(key=lambda x: x[1])
            elif sort_by == "Newest first":
                rows.sort(key=lambda x: x[1], reverse=True)
            elif sort_by == "Most turns":
                rows.sort(key=lambda x: x[2], reverse=True)
            elif sort_by == "Fewest turns":
                rows.sort(key=lambda x: x[2])

            total = len(rows)
            pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page_rows = rows[:PAGE_SIZE]

            return (
                page_rows,
                1,  # current_page
                pages,  # total_pages
                rows,  # filtered_data (all rows)
                f"Page 1 of {pages} ({total:,} conversations)"
            )

        def go_to_page(page, pages, all_rows):
            """Navigate to a specific page."""
            page = max(1, min(page, pages))
            start = (page - 1) * PAGE_SIZE
            end = start + PAGE_SIZE
            page_rows = all_rows[start:end]
            total = len(all_rows)
            return (
                page_rows,
                page,
                f"Page {page} of {pages} ({total:,} conversations)"
            )

        def prev_page(page, pages, all_rows):
            """Go to previous page."""
            return go_to_page(page - 1, pages, all_rows)

        def next_page(page, pages, all_rows):
            """Go to next page."""
            return go_to_page(page + 1, pages, all_rows)

        def view_conversation(evt: gr.SelectData, data):
            if evt.index[0] is None:
                return "", "Select a conversation", None

            row = data.iloc[evt.index[0]].tolist()
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
            inputs=[search_box, filter_extracted, sort_dropdown],
            outputs=[conversations_table, current_page, total_pages, filtered_data, page_info]
        )

        prev_btn.click(
            fn=prev_page,
            inputs=[current_page, total_pages, filtered_data],
            outputs=[conversations_table, current_page, page_info]
        )

        next_btn.click(
            fn=next_page,
            inputs=[current_page, total_pages, filtered_data],
            outputs=[conversations_table, current_page, page_info]
        )

        conversations_table.select(
            fn=view_conversation,
            inputs=[conversations_table],
            outputs=[conv_id_display, messages_display, extraction_display]
        )

    # Return tab, load function, inputs, and outputs for auto-refresh
    def load_conversations_default():
        return load_conversations("", False, "Oldest first")

    return (tab, load_conversations_default, [
        conversations_table, current_page, total_pages, filtered_data, page_info
    ])


# =============================================================================
# PROJECT IDEAS TAB
# =============================================================================

def create_ideas_tab():
    """Create the project ideas tab."""
    with gr.Tab("Ideas") as tab:
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

        load_ideas_btn = gr.Button("Load Ideas", variant="primary")

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

    def load_ideas_default():
        return load_ideas("All", "")

    return (tab, load_ideas_default, [ideas_table, ideas_chart])


# =============================================================================
# PROBLEMS TAB
# =============================================================================

def create_problems_tab():
    """Create the problems tab."""
    with gr.Tab("Problems") as tab:
        gr.Markdown("## Problems & Pain Points")

        search_problems = gr.Textbox(label="Search", placeholder="Search problems...")

        problems_table = gr.Dataframe(
            headers=["Problem", "Context", "Source", "Date"],
            label="",
            interactive=False
        )

        load_problems_btn = gr.Button("Load Problems", variant="primary")

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

    def load_problems_default():
        return load_problems("")

    return (tab, load_problems_default, [problems_table])


# =============================================================================
# TOOLS TAB
# =============================================================================

def create_tools_tab():
    """Create the tools explorer tab."""
    with gr.Tab("Tools") as tab:
        gr.Markdown("## Tools & Technologies")

        tools_table = gr.Dataframe(
            headers=["Tool", "Mentions"],
            label="",
            interactive=False
        )

        load_tools_btn = gr.Button("Load Tools", variant="primary")

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

    return (tab, load_tools, [tools_table, tools_chart])


# =============================================================================
# EMOTIONS TAB
# =============================================================================

def create_emotions_tab():
    """Create the emotions/tone tab."""
    with gr.Tab("Emotions") as tab:
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

        load_emotions_btn = gr.Button("Load Emotions", variant="primary")

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

    def load_emotions_default():
        return load_emotions("All")

    return (tab, load_emotions_default, [emotions_chart, emotions_table])


# =============================================================================
# CONSOLIDATED TAB
# =============================================================================

def load_consolidated_data() -> dict | None:
    """Load consolidated data from disk."""
    if not CONSOLIDATED_FILE.exists():
        return None
    with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_consolidated_tab():
    """Create the consolidated insights tab."""
    with gr.Tab("Consolidation") as tab:
        gr.Markdown("## Consolidated Insights")
        gr.Markdown("*Deduplicated ideas and problems across all conversations*")

        # Summary stats
        with gr.Row():
            total_ideas = gr.Number(label="Unique Ideas", interactive=False)
            total_problems = gr.Number(label="Unique Problems", interactive=False)
            total_workflows = gr.Number(label="Unique Workflows", interactive=False)
            source_count = gr.Number(label="Source Extractions", interactive=False)

        # Tabs for different cluster types
        with gr.Tabs():
            with gr.Tab("Idea Clusters"):
                ideas_table = gr.Dataframe(
                    headers=["Name", "Occurrences", "Date Range", "Description"],
                    label="Consolidated Ideas",
                    interactive=False
                )

                gr.Markdown("### Selected Idea Details")
                idea_detail = gr.JSON(label="Full Details")

            with gr.Tab("Problem Clusters"):
                problems_table = gr.Dataframe(
                    headers=["Name", "Occurrences", "Date Range", "Description"],
                    label="Consolidated Problems",
                    interactive=False
                )

                gr.Markdown("### Selected Problem Details")
                problem_detail = gr.JSON(label="Full Details")

            with gr.Tab("Tools Frequency"):
                tools_chart = gr.Plot(label="Tool Frequency")
                tools_freq_table = gr.Dataframe(
                    headers=["Tool", "Mentions"],
                    label="All Tools",
                    interactive=False
                )

            with gr.Tab("Emotional Timeline"):
                timeline_chart = gr.Plot(label="Emotional Timeline")
                timeline_table = gr.Dataframe(
                    headers=["Date", "Title", "Tone", "Notes"],
                    label="",
                    interactive=False
                )

        with gr.Row():
            refresh_btn = gr.Button("Refresh Consolidated Data", variant="primary")
            run_consolidation_btn = gr.Button("Run Consolidation", variant="primary")

        consolidation_log = gr.Textbox(
            label="Consolidation Log",
            value="Click 'Run Consolidation' to start...",
            lines=8,
            interactive=False
        )

        def load_consolidated():
            data = load_consolidated_data()
            if not data:
                empty_msg = "No consolidated data found. Run consolidation first."
                return (
                    0, 0, 0, 0,
                    [[empty_msg, "", "", ""]],
                    None,
                    [[empty_msg, "", "", ""]],
                    None,
                    go.Figure(),
                    [],
                    go.Figure(),
                    []
                )

            # Stats
            n_ideas = len(data.get('idea_clusters', []))
            n_problems = len(data.get('problem_clusters', []))
            n_workflows = len(data.get('workflow_clusters', []))
            n_sources = data.get('metadata', {}).get('source_extractions', 0)

            # Sort idea clusters by occurrences (highest first)
            sorted_ideas = sorted(
                data.get('idea_clusters', []),
                key=lambda x: x.get('occurrences', 1),
                reverse=True
            )

            # Ideas table
            ideas_rows = []
            for cluster in sorted_ideas:
                date_range = cluster.get('date_range', ['', ''])
                if isinstance(date_range, list) and len(date_range) >= 2:
                    date_str = f"{date_range[0]} - {date_range[1]}"
                else:
                    date_str = str(date_range)
                ideas_rows.append([
                    cluster.get('name', 'Unknown')[:50],
                    cluster.get('occurrences', 1),
                    date_str,
                    cluster.get('description', '')[:100]
                ])

            # Sort problem clusters by occurrences (highest first)
            sorted_problems = sorted(
                data.get('problem_clusters', []),
                key=lambda x: x.get('occurrences', 1),
                reverse=True
            )

            # Problems table
            problems_rows = []
            for cluster in sorted_problems:
                date_range = cluster.get('date_range', ['', ''])
                if isinstance(date_range, list) and len(date_range) >= 2:
                    date_str = f"{date_range[0]} - {date_range[1]}"
                else:
                    date_str = str(date_range)
                problems_rows.append([
                    cluster.get('name', 'Unknown')[:50],
                    cluster.get('occurrences', 1),
                    date_str,
                    cluster.get('description', '')[:100]
                ])

            # Tools chart
            tool_freq = data.get('tool_frequency', {})
            sorted_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)
            top_20 = sorted_tools[:20]

            if top_20:
                tools_fig = px.bar(
                    x=[t[0] for t in top_20],
                    y=[t[1] for t in top_20],
                    title="Top 20 Tools",
                    labels={'x': 'Tool', 'y': 'Mentions'}
                )
                tools_fig.update_traces(marker_color='#00ff00')
                tools_fig.update_xaxes(tickangle=45)
                apply_matrix_theme(tools_fig)
            else:
                tools_fig = go.Figure()
                apply_matrix_theme(tools_fig)

            tools_rows = [[t[0], t[1]] for t in sorted_tools]

            # Emotional timeline
            timeline = data.get('emotional_timeline', [])
            timeline_rows = []
            for entry in timeline:
                timeline_rows.append([
                    entry.get('date', ''),
                    entry.get('title', '')[:40],
                    entry.get('tone', ''),
                    entry.get('notes', '')[:60]
                ])

            # Timeline chart - tone over time
            if timeline:
                tone_mapping = {'excited': 2, 'curious': 1, 'neutral': 0, 'frustrated': -1, 'stuck': -2}
                dates = [e.get('date', '') for e in timeline]
                tone_values = [tone_mapping.get(e.get('tone', 'neutral'), 0) for e in timeline]

                timeline_fig = go.Figure()
                timeline_fig.add_trace(go.Scatter(
                    x=dates,
                    y=tone_values,
                    mode='lines+markers',
                    marker=dict(color='#00ff00', size=8),
                    line=dict(color='#00ff00', width=2),
                    hovertext=[e.get('title', '') for e in timeline]
                ))
                timeline_fig.update_layout(
                    title="Emotional Timeline",
                    yaxis=dict(
                        ticktext=['stuck', 'frustrated', 'neutral', 'curious', 'excited'],
                        tickvals=[-2, -1, 0, 1, 2]
                    )
                )
                apply_matrix_theme(timeline_fig)
            else:
                timeline_fig = go.Figure()
                apply_matrix_theme(timeline_fig)

            return (
                n_ideas, n_problems, n_workflows, n_sources,
                ideas_rows,
                sorted_ideas,  # Pass sorted data so row index matches
                problems_rows,
                sorted_problems,  # Pass sorted data so row index matches
                tools_fig,
                tools_rows,
                timeline_fig,
                timeline_rows
            )

        def run_consolidation():
            from datetime import datetime
            import subprocess
            import time

            STATUS_FILE = Path("data/consolidation_status.json")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            yield f"[{timestamp}] Starting consolidation...\n\nThis may take several minutes. Please wait..."

            # Clear old status file
            if STATUS_FILE.exists():
                STATUS_FILE.unlink()

            try:
                # Start process in background
                process = subprocess.Popen(
                    [sys.executable, "consolidate.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                    text=True
                )

                start_time = time.time()
                last_message = ""

                # Poll for status updates
                while process.poll() is None:  # Process still running
                    if STATUS_FILE.exists():
                        try:
                            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                                status = json.load(f)
                            message = status.get('message', '')
                            progress = status.get('progress', 0)
                            elapsed = status.get('elapsed_seconds', 0)
                            eta = status.get('eta_seconds')

                            if message != last_message or True:  # Always update for time
                                last_message = message
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                elapsed_str = format_time(elapsed)
                                eta_str = format_time(eta) if eta else "calculating..."
                                progress_bar = f"[{'█' * int(progress/5)}{'░' * (20-int(progress/5))}] {progress:.0f}%"
                                yield f"[{ts}] {message}\nElapsed: {elapsed_str} | ETA: {eta_str}\n{progress_bar}"

                            if status.get('complete') or status.get('error'):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass  # Status file being written

                    time.sleep(2)  # Poll every 2 seconds

                # Get final output
                stdout, stderr = process.communicate(timeout=10)
                end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Check final status
                if STATUS_FILE.exists():
                    try:
                        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        elapsed = status.get('elapsed_seconds', 0)
                        elapsed_str = format_time(elapsed)
                        if status.get('error'):
                            yield f"[{end_timestamp}] {status.get('message', 'Unknown error')}"
                            return
                        if status.get('complete'):
                            yield f"[{end_timestamp}] {status.get('message', 'Consolidation complete.')}\nTotal time: {elapsed_str}\n\nClick 'Refresh Consolidated Data' to see results."
                            return
                    except (json.JSONDecodeError, IOError):
                        pass

                if process.returncode == 0:
                    yield f"[{end_timestamp}] Consolidation complete.\n\nClick 'Refresh Consolidated Data' to see results."
                else:
                    yield f"[{end_timestamp}] Consolidation finished with errors.\n\n{stderr}"

            except Exception as e:
                yield f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}"

        def view_idea_detail(evt: gr.SelectData, ideas_data):
            if not ideas_data or evt.index[0] is None:
                return None
            if evt.index[0] < len(ideas_data):
                return ideas_data[evt.index[0]]
            return None

        def view_problem_detail(evt: gr.SelectData, problems_data):
            if not problems_data or evt.index[0] is None:
                return None
            if evt.index[0] < len(problems_data):
                return problems_data[evt.index[0]]
            return None

        # Store full data for detail views
        ideas_data_store = gr.State([])
        problems_data_store = gr.State([])

        refresh_btn.click(
            fn=load_consolidated,
            outputs=[
                total_ideas, total_problems, total_workflows, source_count,
                ideas_table, ideas_data_store,
                problems_table, problems_data_store,
                tools_chart, tools_freq_table,
                timeline_chart, timeline_table
            ]
        )

        run_consolidation_btn.click(
            fn=run_consolidation,
            outputs=[consolidation_log]
        )

        ideas_table.select(
            fn=view_idea_detail,
            inputs=[ideas_data_store],
            outputs=[idea_detail]
        )

        problems_table.select(
            fn=view_problem_detail,
            inputs=[problems_data_store],
            outputs=[problem_detail]
        )

    return (tab, load_consolidated, [
        total_ideas, total_problems, total_workflows, source_count,
        ideas_table, ideas_data_store,
        problems_table, problems_data_store,
        tools_chart, tools_freq_table,
        timeline_chart, timeline_table
    ])


# =============================================================================
# CATEGORIES TAB
# =============================================================================

CATEGORIZED_FILE = CONSOLIDATED_DIR / "categorized.json"


def load_categorized_data() -> dict | None:
    """Load categorized data from disk."""
    if not CATEGORIZED_FILE.exists():
        return None
    with open(CATEGORIZED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_categories_tab():
    """Create the categorization and scoring tab."""
    with gr.Tab("Categorization") as tab:
        gr.Markdown("## Categorized Ideas")
        gr.Markdown("*Scored and prioritized by effort, monetization, utility, passion, and recurrence*")

        # Summary stats
        with gr.Row():
            quick_wins_count = gr.Number(label="Quick Wins", interactive=False)
            validate_count = gr.Number(label="Validate", interactive=False)
            revive_count = gr.Number(label="Revive", interactive=False)
            someday_count = gr.Number(label="Someday", interactive=False)

        # Category filter
        category_filter = gr.Radio(
            choices=["All", "quick_win", "validate", "revive", "someday"],
            value="All",
            label="Filter by Category"
        )

        # Main table with scores
        categories_table = gr.Dataframe(
            headers=["Name", "Category", "Score", "Effort", "Monetization", "Utility", "Passion", "Recurrence"],
            label="Scored Ideas",
            interactive=False
        )

        gr.Markdown("### Selected Idea Details")
        idea_detail = gr.JSON(label="Full Details")

        with gr.Row():
            refresh_btn = gr.Button("Refresh Categories", variant="primary")
            run_categorization_btn = gr.Button("Run Categorization", variant="primary")

        categorization_log = gr.Textbox(
            label="Categorization Log",
            value="Click 'Run Categorization' to start...",
            lines=8,
            interactive=False
        )

        # Quick Wins summary
        gr.Markdown("---")
        gr.Markdown("### Quick Wins")
        gr.Markdown("*Low effort + High utility + High passion*")
        quick_wins_table = gr.Dataframe(
            headers=["Name", "Description", "Why It's a Quick Win"],
            label="",
            interactive=False
        )

        def load_categories(filter_category):
            data = load_categorized_data()
            if not data:
                empty_msg = "No categorized data found. Run categorization first."
                return (
                    0, 0, 0, 0,
                    [[empty_msg, "", "", "", "", "", "", ""]],
                    None,
                    [[empty_msg, "", ""]]
                )

            # Category counts
            counts = data.get('metadata', {}).get('category_counts', {})
            n_quick = counts.get('quick_win', 0)
            n_validate = counts.get('validate', 0)
            n_revive = counts.get('revive', 0)
            n_someday = counts.get('someday', 0)

            # Build main table
            ideas = data.get('ideas', [])
            rows = []

            for idea in ideas:
                category = idea.get('category', 'someday')

                # Apply filter
                if filter_category != "All" and category != filter_category:
                    continue

                scores = idea.get('scores', {})
                rows.append([
                    idea.get('name', 'Unknown')[:40],
                    category,
                    idea.get('composite_score', 0),
                    scores.get('effort', '-'),
                    scores.get('monetization', '-'),
                    scores.get('personal_utility', '-'),
                    scores.get('passion', '-'),
                    scores.get('recurrence', '-')
                ])

            # Quick wins table
            quick_wins = data.get('by_category', {}).get('quick_win', [])
            quick_rows = []
            for idea in quick_wins:
                scores = idea.get('scores', {})
                quick_rows.append([
                    idea.get('name', 'Unknown'),
                    idea.get('description', '')[:100],
                    scores.get('llm_reasoning', '')[:100]
                ])

            if not quick_rows:
                quick_rows = [["No quick wins identified", "", ""]]

            return (
                n_quick, n_validate, n_revive, n_someday,
                rows,
                ideas,
                quick_rows
            )

        def run_categorization():
            from datetime import datetime
            import subprocess
            import time

            STATUS_FILE = Path("data/categorization_status.json")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            yield f"[{timestamp}] Starting categorization...\n\nThis may take a minute or two. Please wait..."

            # Clear old status file
            if STATUS_FILE.exists():
                STATUS_FILE.unlink()

            try:
                # Start process in background
                process = subprocess.Popen(
                    [sys.executable, "categorize.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                    text=True
                )

                start_time = time.time()
                last_message = ""

                # Poll for status updates
                while process.poll() is None:  # Process still running
                    if STATUS_FILE.exists():
                        try:
                            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                                status = json.load(f)
                            message = status.get('message', '')
                            progress = status.get('progress', 0)
                            elapsed = status.get('elapsed_seconds', 0)
                            eta = status.get('eta_seconds')

                            if message != last_message or True:  # Always update for time
                                last_message = message
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                elapsed_str = format_time(elapsed)
                                eta_str = format_time(eta) if eta else "calculating..."
                                progress_bar = f"[{'█' * int(progress/5)}{'░' * (20-int(progress/5))}] {progress:.0f}%"
                                yield f"[{ts}] {message}\nElapsed: {elapsed_str} | ETA: {eta_str}\n{progress_bar}"

                            if status.get('complete') or status.get('error'):
                                break
                        except (json.JSONDecodeError, IOError):
                            pass  # Status file being written

                    # Timeout after 10 minutes
                    if time.time() - start_time > 600:
                        process.kill()
                        yield f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Process timed out after 10 minutes."
                        return

                    time.sleep(2)  # Poll every 2 seconds

                # Get final output
                stdout, stderr = process.communicate(timeout=10)
                end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Check final status
                if STATUS_FILE.exists():
                    try:
                        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        elapsed = status.get('elapsed_seconds', 0)
                        elapsed_str = format_time(elapsed)
                        if status.get('error'):
                            yield f"[{end_timestamp}] {status.get('message', 'Unknown error')}"
                            return
                        if status.get('complete'):
                            yield f"[{end_timestamp}] {status.get('message', 'Categorization complete.')}\nTotal time: {elapsed_str}\n\nClick 'Refresh Categories' to see results."
                            return
                    except (json.JSONDecodeError, IOError):
                        pass

                if process.returncode == 0:
                    yield f"[{end_timestamp}] Categorization complete.\n\nClick 'Refresh Categories' to see results."
                else:
                    yield f"[{end_timestamp}] Categorization finished with errors.\n\n{stderr}"

            except Exception as e:
                yield f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}"

        def view_idea_detail(evt: gr.SelectData, ideas_data):
            if not ideas_data or evt.index[0] is None:
                return None
            if evt.index[0] < len(ideas_data):
                return ideas_data[evt.index[0]]
            return None

        # Store full data for detail view
        ideas_data_store = gr.State([])

        refresh_btn.click(
            fn=load_categories,
            inputs=[category_filter],
            outputs=[
                quick_wins_count, validate_count, revive_count, someday_count,
                categories_table, ideas_data_store, quick_wins_table
            ]
        )

        category_filter.change(
            fn=load_categories,
            inputs=[category_filter],
            outputs=[
                quick_wins_count, validate_count, revive_count, someday_count,
                categories_table, ideas_data_store, quick_wins_table
            ]
        )

        run_categorization_btn.click(
            fn=run_categorization,
            outputs=[categorization_log]
        )

        categories_table.select(
            fn=view_idea_detail,
            inputs=[ideas_data_store],
            outputs=[idea_detail]
        )

    def load_categories_default():
        return load_categories("All")

    return (tab, load_categories_default, [
        quick_wins_count, validate_count, revive_count, someday_count,
        categories_table, ideas_data_store, quick_wins_table
    ])


# =============================================================================
# SYNTHESIS TAB (Phase 5)
# =============================================================================

def create_synthesis_tab():
    """Create the creative synthesis tab for generating novel project ideas."""
    from synthesizer import (
        run_synthesis, load_generated_ideas, load_passion_profile,
        save_idea, dismiss_idea, develop_idea_further, get_synthesis_status,
        load_saved_ideas, get_developed_ideas, get_developed_spec
    )

    with gr.Tab("Synthesis") as tab:
        gr.Markdown("## Creative Synthesis Engine")
        gr.Markdown("Generate novel project ideas based on patterns in your conversation history.")

        # Status and controls
        with gr.Row():
            with gr.Column(scale=2):
                generate_btn = gr.Button("Generate Projects", variant="primary", size="lg")
            with gr.Column(scale=3):
                synthesis_status = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")

        synthesis_progress = gr.Textbox(
            label="Progress",
            value="[░░░░░░░░░░░░░░░░░░░░] 0%",
            interactive=False,
            lines=1
        )

        gr.Markdown("---")

        # Profile section
        with gr.Accordion("Your Passion Profile", open=False):
            profile_summary = gr.Markdown("*Run synthesis to generate your passion profile*")
            profile_json = gr.JSON(label="Full Profile", visible=False)

        gr.Markdown("---")
        gr.Markdown("### Generated Projects")

        # Ideas by strategy in nested tabs
        with gr.Tabs():
            with gr.Tab("Passion Intersections"):
                gr.Markdown("*Novel combinations of your top themes*")
                intersection_df = gr.Dataframe(
                    headers=["Name", "Description", "Themes", "Score"],
                    label="Intersection Projects",
                    wrap=True
                )

            with gr.Tab("Problem Solvers"):
                gr.Markdown("*Practical solutions to your pain points*")
                solution_df = gr.Dataframe(
                    headers=["Name", "Problem", "Tools", "Score"],
                    label="Solution Projects",
                    wrap=True
                )

            with gr.Tab("Profile Matches"):
                gr.Markdown("*Projects tailored to your overall patterns*")
                profile_df = gr.Dataframe(
                    headers=["Name", "Description", "Alignment", "Score"],
                    label="Profile Projects",
                    wrap=True
                )

            with gr.Tab("Time Capsules"):
                gr.Markdown("*Forgotten gems resurfaced with fresh perspective*")
                capsule_df = gr.Dataframe(
                    headers=["Name", "Original", "Months Ago", "Letter", "Score"],
                    label="Time Capsule Projects",
                    wrap=True
                )

            with gr.Tab("All Projects"):
                gr.Markdown("*All generated projects sorted by score*")
                all_ideas_df = gr.Dataframe(
                    headers=["Name", "Strategy", "Description", "Score"],
                    label="All Projects",
                    wrap=True
                )

            with gr.Tab("Saved Projects"):
                gr.Markdown("*Projects you've saved for later*")
                saved_ideas_df = gr.Dataframe(
                    headers=["Name", "Description", "Strategy", "Score", "Saved At"],
                    label="Saved Projects",
                    wrap=True
                )
                refresh_saved_btn = gr.Button("Refresh Saved Projects", size="sm", variant="secondary")

            with gr.Tab("Developed Specs"):
                gr.Markdown("*Project specifications you've developed*")
                developed_list_dropdown = gr.Dropdown(
                    label="Select Developed Project",
                    choices=[],
                    interactive=True
                )
                refresh_developed_btn = gr.Button("Refresh List", size="sm", variant="secondary")
                developed_spec_json = gr.JSON(label="Project Specification")

        gr.Markdown("---")
        gr.Markdown("### Project Actions")

        with gr.Row():
            idea_dropdown = gr.Dropdown(
                label="Select Project",
                choices=[],
                interactive=True
            )
            save_btn = gr.Button("Save", variant="secondary")
            dismiss_btn = gr.Button("Dismiss", variant="secondary")
            develop_btn = gr.Button("Develop Further", variant="primary")

        action_result = gr.Markdown("")

        with gr.Accordion("Developed Specification", open=False, visible=False) as dev_accordion:
            developed_output = gr.JSON(label="Project Specification")

        # Helper functions
        def format_ideas_for_display(ideas: list, strategy: str = None) -> list:
            """Format ideas for dataframe display."""
            rows = []
            for idea in ideas:
                if strategy and idea.get("strategy") != strategy:
                    continue

                name = idea.get("name", "Unknown")
                desc = idea.get("description", "")[:100] + "..." if len(idea.get("description", "")) > 100 else idea.get("description", "")
                score = idea.get("composite_score", 0)

                if strategy == "intersection":
                    themes = ", ".join(idea.get("themes_combined", []))[:50]
                    rows.append([name, desc, themes, score])
                elif strategy == "problem_solution":
                    problem = idea.get("problem_addressed", "")[:50]
                    tools = ", ".join(idea.get("tools_used", []))[:30]
                    rows.append([name, problem, tools, score])
                elif strategy == "profile_based":
                    align = idea.get("profile_alignment", "")[:50]
                    rows.append([name, desc, align, score])
                elif strategy == "time_capsule":
                    orig = idea.get("original_idea", "")[:30]
                    months = idea.get("months_ago", 0)
                    letter = idea.get("letter_from_past", "")[:50]
                    rows.append([name, orig, months, letter, score])
                else:
                    strat = idea.get("strategy", "unknown")
                    rows.append([name, strat, desc, score])

            return rows

        def run_synthesis_handler():
            """Handle synthesis generation."""
            try:
                result = run_synthesis()

                if not result:
                    return (
                        "Synthesis failed - check console for errors",
                        "[░░░░░░░░░░░░░░░░░░░░] 0%",
                        "*Generation failed*",
                        gr.update(visible=False),
                        [], [], [], [], [],
                        gr.update(choices=[])
                    )

                profile = result.get("profile", {})
                all_ideas = result.get("all_ideas", [])

                # Format profile summary
                summary = profile.get("summary", "No summary available")
                themes = [t.get("theme", t) if isinstance(t, dict) else t for t in profile.get("core_themes", [])]
                profile_md = f"**Summary:** {summary}\n\n**Core Themes:** {', '.join(themes[:5])}"

                # Format dataframes
                intersection_rows = format_ideas_for_display(all_ideas, "intersection")
                solution_rows = format_ideas_for_display(all_ideas, "problem_solution")
                profile_rows = format_ideas_for_display(all_ideas, "profile_based")
                capsule_rows = format_ideas_for_display(all_ideas, "time_capsule")
                all_rows = format_ideas_for_display(all_ideas)

                # Build dropdown choices
                choices = [f"{idea.get('name', 'Unknown')} ({idea.get('id', '')})" for idea in all_ideas]

                return (
                    f"Complete! Generated {len(all_ideas)} projects",
                    "[████████████████████] 100%",
                    profile_md,
                    gr.update(value=profile, visible=True),
                    intersection_rows,
                    solution_rows,
                    profile_rows,
                    capsule_rows,
                    all_rows,
                    gr.update(choices=choices, value=choices[0] if choices else None)
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return (
                    f"Error: {str(e)}",
                    "[░░░░░░░░░░░░░░░░░░░░] 0%",
                    "*Error during generation*",
                    gr.update(visible=False),
                    [], [], [], [], [],
                    gr.update(choices=[])
                )

        def refresh_display():
            """Refresh display from saved files."""
            try:
                generated = load_generated_ideas()
                profile = load_passion_profile()
                all_ideas = generated.get("ideas", [])

                if not all_ideas:
                    return (
                        "No generated projects found",
                        "[░░░░░░░░░░░░░░░░░░░░] 0%",
                        "*Run synthesis to generate projects*",
                        gr.update(visible=False),
                        [], [], [], [], [],
                        gr.update(choices=[])
                    )

                # Format profile summary
                summary = profile.get("summary", "No profile available")
                themes = [t.get("theme", t) if isinstance(t, dict) else t for t in profile.get("core_themes", [])]
                profile_md = f"**Summary:** {summary}\n\n**Core Themes:** {', '.join(themes[:5])}"

                # Format dataframes
                intersection_rows = format_ideas_for_display(all_ideas, "intersection")
                solution_rows = format_ideas_for_display(all_ideas, "problem_solution")
                profile_rows = format_ideas_for_display(all_ideas, "profile_based")
                capsule_rows = format_ideas_for_display(all_ideas, "time_capsule")
                all_rows = format_ideas_for_display(all_ideas)

                choices = [f"{idea.get('name', 'Unknown')} ({idea.get('id', '')})" for idea in all_ideas]

                return (
                    f"Loaded {len(all_ideas)} projects",
                    "[████████████████████] 100%",
                    profile_md,
                    gr.update(value=profile, visible=True),
                    intersection_rows,
                    solution_rows,
                    profile_rows,
                    capsule_rows,
                    all_rows,
                    gr.update(choices=choices, value=choices[0] if choices else None)
                )
            except Exception as e:
                return (
                    f"Error loading: {str(e)}",
                    "[░░░░░░░░░░░░░░░░░░░░] 0%",
                    "*Error loading data*",
                    gr.update(visible=False),
                    [], [], [], [], [],
                    gr.update(choices=[])
                )

        def save_idea_handler(selected):
            """Handle saving a project."""
            if not selected:
                return "No project selected"
            idea_id = selected.split("(")[-1].rstrip(")")
            if save_idea(idea_id):
                return f"Saved: {selected.split('(')[0].strip()}"
            return "Failed to save project"

        def dismiss_idea_handler(selected):
            """Handle dismissing a project."""
            if not selected:
                return "No project selected"
            idea_id = selected.split("(")[-1].rstrip(")")
            if dismiss_idea(idea_id):
                return f"Dismissed: {selected.split('(')[0].strip()}"
            return "Failed to dismiss project"

        def develop_idea_handler(selected):
            """Handle developing a project further."""
            if not selected:
                return "No project selected", gr.update(visible=False), None
            idea_id = selected.split("(")[-1].rstrip(")")
            spec = develop_idea_further(idea_id)
            if spec:
                return (
                    f"Developed: {selected.split('(')[0].strip()}",
                    gr.update(visible=True),
                    spec
                )
            return "Failed to develop project", gr.update(visible=False), None

        def refresh_saved_ideas_handler():
            """Refresh the saved ideas display."""
            saved = load_saved_ideas()
            if not saved:
                return []

            rows = []
            for idea in saved:
                name = idea.get("name", "Unknown")
                desc = idea.get("description", "")[:80] + "..." if len(idea.get("description", "")) > 80 else idea.get("description", "")
                strategy = idea.get("strategy", "unknown")
                score = idea.get("composite_score", 0)
                saved_at = idea.get("saved_at", "Unknown")[:10]  # Just date part
                rows.append([name, desc, strategy, score, saved_at])
            return rows

        def refresh_developed_list_handler():
            """Refresh the developed ideas dropdown."""
            developed = get_developed_ideas()
            if not developed:
                return gr.update(choices=[], value=None), None

            choices = [f"{d['idea_name']} ({d['idea_id']})" for d in developed]
            return gr.update(choices=choices, value=choices[0] if choices else None), None

        def load_developed_spec_handler(selected):
            """Load and display a developed specification."""
            if not selected:
                return None
            idea_id = selected.split("(")[-1].rstrip(")")
            spec = get_developed_spec(idea_id)
            return spec

        # Wire up events
        generate_btn.click(
            fn=run_synthesis_handler,
            outputs=[
                synthesis_status, synthesis_progress, profile_summary, profile_json,
                intersection_df, solution_df, profile_df, capsule_df, all_ideas_df,
                idea_dropdown
            ]
        )

        refresh_btn.click(
            fn=refresh_display,
            outputs=[
                synthesis_status, synthesis_progress, profile_summary, profile_json,
                intersection_df, solution_df, profile_df, capsule_df, all_ideas_df,
                idea_dropdown
            ]
        )

        save_btn.click(
            fn=save_idea_handler,
            inputs=[idea_dropdown],
            outputs=[action_result]
        )

        dismiss_btn.click(
            fn=dismiss_idea_handler,
            inputs=[idea_dropdown],
            outputs=[action_result]
        )

        develop_btn.click(
            fn=develop_idea_handler,
            inputs=[idea_dropdown],
            outputs=[action_result, dev_accordion, developed_output]
        )

        # Wire up saved/developed tabs
        refresh_saved_btn.click(
            fn=refresh_saved_ideas_handler,
            outputs=[saved_ideas_df]
        )

        refresh_developed_btn.click(
            fn=refresh_developed_list_handler,
            outputs=[developed_list_dropdown, developed_spec_json]
        )

        developed_list_dropdown.change(
            fn=load_developed_spec_handler,
            inputs=[developed_list_dropdown],
            outputs=[developed_spec_json]
        )

    return (tab, refresh_display, [
        synthesis_status, synthesis_progress, profile_summary, profile_json,
        intersection_df, solution_df, profile_df, capsule_df, all_ideas_df,
        idea_dropdown
    ])


# =============================================================================
# SETTINGS TAB
# =============================================================================

def create_settings_tab():
    """Create the settings and data management tab."""
    from config import load_config, save_config
    from data_management import get_data_status, reset_extractions, reset_all_processed, format_status_markdown
    from llm_provider import check_ollama_connection, list_ollama_models

    with gr.Tab("Settings"):
        gr.Markdown("## Settings")

        # =====================================================================
        # API Configuration Section
        # =====================================================================
        gr.Markdown("### API Configuration")

        config = load_config()
        current_provider = config.get("api_provider", "openai")
        is_ollama = current_provider == "ollama"

        with gr.Row():
            with gr.Column():
                provider_dropdown = gr.Dropdown(
                    choices=["openai", "anthropic", "ollama"],
                    value=current_provider,
                    label="LLM Provider"
                )

        # Cloud provider settings (OpenAI/Anthropic)
        with gr.Group(visible=not is_ollama) as cloud_settings:
            with gr.Row():
                with gr.Column():
                    model_input = gr.Textbox(
                        value=config.get("model", "gpt-4o-mini"),
                        label="Model"
                    )
                with gr.Column():
                    api_key_input = gr.Textbox(
                        value=config.get("api_key", ""),
                        label="API Key",
                        type="password",
                        placeholder="Leave blank to use environment variable"
                    )

        # Ollama settings (Local)
        with gr.Group(visible=is_ollama) as ollama_settings:
            with gr.Row():
                with gr.Column():
                    ollama_host_input = gr.Textbox(
                        value=config.get("ollama_host", "http://localhost:11434"),
                        label="Ollama Host"
                    )
                with gr.Column():
                    # Get initial Ollama models if connected
                    initial_models = []
                    initial_status = "Checking connection..."
                    if is_ollama:
                        try:
                            if check_ollama_connection(config.get("ollama_host", "http://localhost:11434")):
                                initial_models = list_ollama_models(config.get("ollama_host", "http://localhost:11434"))
                                initial_status = "Connected to Ollama"
                            else:
                                initial_status = "Cannot connect to Ollama. Is it running?"
                        except Exception:
                            initial_status = "Cannot connect to Ollama. Is it running?"

                    ollama_model_dropdown = gr.Dropdown(
                        choices=initial_models,
                        value=config.get("ollama_model", "") or (initial_models[0] if initial_models else None),
                        label="Ollama Model"
                    )

            with gr.Row():
                refresh_models_btn = gr.Button("Refresh Models", size="sm")
                ollama_status = gr.Markdown(initial_status)

        # Rate limit (shared)
        with gr.Row():
            rate_limit_input = gr.Number(
                label="Requests per minute (1-60)",
                value=config.get("requests_per_minute", 20),
                minimum=1,
                maximum=60,
                step=1
            )

        save_settings_btn = gr.Button("Save Settings", variant="primary")
        settings_result = gr.Markdown("")

        # Provider change handler - also refreshes models when switching to Ollama
        def on_provider_change(provider, host):
            is_ollama = provider == "ollama"
            if is_ollama:
                # Fetch models when switching to Ollama
                try:
                    if check_ollama_connection(host):
                        models = list_ollama_models(host)
                        return (
                            gr.update(visible=False),  # cloud_settings
                            gr.update(visible=True),   # ollama_settings
                            gr.update(choices=models, value=models[0] if models else None),
                            "Connected to Ollama"
                        )
                    else:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(choices=[], value=None),
                            "Cannot connect to Ollama. Is it running?"
                        )
                except Exception as e:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(choices=[], value=None),
                        f"Error: {e}"
                    )
            else:
                return (
                    gr.update(visible=True),   # cloud_settings
                    gr.update(visible=False),  # ollama_settings
                    gr.update(),  # no change to dropdown
                    ""  # no change to status
                )

        provider_dropdown.change(
            fn=on_provider_change,
            inputs=[provider_dropdown, ollama_host_input],
            outputs=[cloud_settings, ollama_settings, ollama_model_dropdown, ollama_status]
        )

        # Refresh Ollama models handler
        def refresh_ollama_models_handler(host):
            try:
                if check_ollama_connection(host):
                    models = list_ollama_models(host)
                    if models:
                        return (
                            gr.update(choices=models, value=models[0]),
                            "Connected to Ollama"
                        )
                    else:
                        return (
                            gr.update(choices=[], value=None),
                            "Connected but no models found. Run: ollama pull gemma3:4b"
                        )
                else:
                    return (
                        gr.update(choices=[], value=None),
                        "Cannot connect to Ollama. Is it running?"
                    )
            except Exception as e:
                return (
                    gr.update(choices=[], value=None),
                    f"Error: {e}"
                )

        refresh_models_btn.click(
            fn=refresh_ollama_models_handler,
            inputs=[ollama_host_input],
            outputs=[ollama_model_dropdown, ollama_status]
        )

        # Save settings handler
        def save_settings(provider, model, api_key, rate_limit, ollama_host, ollama_model):
            try:
                current_config = load_config()
                # Update existing config to preserve other settings (like prompts)
                current_config.update({
                    "api_provider": provider,
                    "model": model,
                    "api_key": api_key,
                    "requests_per_minute": int(rate_limit),
                    "retry_attempts": 2,
                    "theme_color": current_config.get("theme_color", "#00ff00"),
                    # Ollama settings
                    "ollama_host": ollama_host,
                    "ollama_model": ollama_model or "",
                    "ollama_timeout": current_config.get("ollama_timeout", 300),
                })
                save_config(current_config)

                # Validate if Ollama is selected
                if provider == "ollama":
                    if not ollama_model:
                        return "Warning: No Ollama model selected. Please select a model."
                    if not check_ollama_connection(ollama_host):
                        return "Warning: Settings saved but cannot connect to Ollama."

                return "Settings saved successfully."
            except Exception as e:
                return f"Error saving settings: {e}"

        save_settings_btn.click(
            fn=save_settings,
            inputs=[provider_dropdown, model_input, api_key_input, rate_limit_input, ollama_host_input, ollama_model_dropdown],
            outputs=[settings_result]
        )

        # =====================================================================
        # Theme Configuration Section
        # =====================================================================
        gr.Markdown("---")
        gr.Markdown("### Theme Color")
        gr.Markdown("*Change the UI accent color. Changes apply live; restart app for full effect.*")

        with gr.Row():
            color_picker = gr.ColorPicker(
                value=config.get("theme_color", "#00ff00"),
                label="Theme Color",
                interactive=True
            )
            save_color_btn = gr.Button("Save Color", variant="primary")

        color_result = gr.Markdown("")

        def save_theme_color(color):
            try:
                # Convert to hex format for consistent storage
                r, g, b = parse_color_to_rgb(color)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                current_config = load_config()
                current_config["theme_color"] = hex_color
                save_config(current_config)
                return f"Theme color saved: {hex_color}"
            except Exception as e:
                return f"Error saving color: {e}"

        # Live color update via JavaScript
        color_picker.change(
            fn=None,
            inputs=[color_picker],
            outputs=None,
            js="(color) => { updateThemeColor(color); return color; }"
        )

        save_color_btn.click(
            fn=save_theme_color,
            inputs=[color_picker],
            outputs=[color_result]
        )

        # =====================================================================
        # Prompt Customization Section
        # =====================================================================
        gr.Markdown("---")
        gr.Markdown("### Prompt Customization")
        gr.Markdown("*Customize LLM prompts used throughout the application. Use Reset to Default if you break something.*")

        from prompts import DEFAULT_PROMPTS, get_prompt, get_default_prompt, get_prompt_metadata

        # Build dropdown choices with grouping labels
        prompt_choices = [
            ("Extraction", "extraction"),
            ("Extraction Retry", "extraction_retry"),
            ("Consolidate Ideas", "consolidate_ideas"),
            ("Consolidate Problems", "consolidate_problems"),
            ("Consolidate Workflows", "consolidate_workflows"),
            ("Idea Scoring", "scoring"),
            ("Passion Profile", "passion_profile"),
            ("Intersection Ideas", "intersection_ideas"),
            ("Solution Ideas", "solution_ideas"),
            ("Profile-Based Ideas", "profile_ideas"),
            ("Time Capsule", "time_capsule"),
            ("Deduplication", "deduplication"),
            ("Generated Ideas Scoring", "generated_scoring"),
            ("Project Development", "project_development"),
        ]

        prompt_dropdown = gr.Dropdown(
            choices=[c[0] for c in prompt_choices],
            value="Extraction",
            label="Select Prompt"
        )

        # Map display name to key
        prompt_name_to_key = {c[0]: c[1] for c in prompt_choices}

        # Get initial prompt
        initial_key = "extraction"
        initial_config = load_config()
        initial_template, initial_system = get_prompt(initial_config, initial_key)
        initial_meta = get_prompt_metadata(initial_key)

        prompt_template_input = gr.Textbox(
            value=initial_template,
            label="Template",
            lines=15,
            max_lines=30,
            placeholder="Enter the prompt template..."
        )

        prompt_variables_display = gr.Markdown(
            f"**Required variables:** `{', '.join(initial_meta.get('variables', []))}`" if initial_meta.get('variables') else "**No variables required**"
        )

        prompt_system_input = gr.Textbox(
            value=initial_system,
            label="System Prompt",
            lines=2,
            placeholder="Enter the system prompt (optional)..."
        )

        with gr.Row():
            reset_prompt_btn = gr.Button("Reset to Default", variant="secondary")
            save_prompt_btn = gr.Button("Save Prompt", variant="primary")

        prompt_result = gr.Markdown("")

        # Handler for dropdown change - load the selected prompt
        def on_prompt_select(prompt_name):
            prompt_key = prompt_name_to_key.get(prompt_name, "extraction")
            current_config = load_config()
            template, system_prompt = get_prompt(current_config, prompt_key)
            meta = get_prompt_metadata(prompt_key)
            variables_text = f"**Required variables:** `{', '.join(meta.get('variables', []))}`" if meta.get('variables') else "**No variables required**"
            return template, system_prompt, variables_text

        prompt_dropdown.change(
            fn=on_prompt_select,
            inputs=[prompt_dropdown],
            outputs=[prompt_template_input, prompt_system_input, prompt_variables_display]
        )

        # Handler for Reset to Default
        def reset_prompt_to_default(prompt_name):
            prompt_key = prompt_name_to_key.get(prompt_name, "extraction")
            try:
                # Remove custom prompt from config
                current_config = load_config()
                if "prompts" in current_config and prompt_key in current_config["prompts"]:
                    del current_config["prompts"][prompt_key]
                    save_config(current_config)

                # Get default prompt
                template, system_prompt = get_default_prompt(prompt_key)
                meta = get_prompt_metadata(prompt_key)
                variables_text = f"**Required variables:** `{', '.join(meta.get('variables', []))}`" if meta.get('variables') else "**No variables required**"
                return template, system_prompt, variables_text, "Prompt reset to default."
            except Exception as e:
                return gr.update(), gr.update(), gr.update(), f"Error resetting prompt: {e}"

        reset_prompt_btn.click(
            fn=reset_prompt_to_default,
            inputs=[prompt_dropdown],
            outputs=[prompt_template_input, prompt_system_input, prompt_variables_display, prompt_result]
        )

        # Handler for Save Prompt
        def save_custom_prompt(prompt_name, template, system_prompt):
            prompt_key = prompt_name_to_key.get(prompt_name, "extraction")
            try:
                current_config = load_config()

                # Initialize prompts dict if needed
                if "prompts" not in current_config:
                    current_config["prompts"] = {}

                # Save custom prompt
                current_config["prompts"][prompt_key] = {
                    "template": template,
                    "system_prompt": system_prompt
                }
                save_config(current_config)
                return "Prompt saved successfully."
            except Exception as e:
                return f"Error saving prompt: {e}"

        save_prompt_btn.click(
            fn=save_custom_prompt,
            inputs=[prompt_dropdown, prompt_template_input, prompt_system_input],
            outputs=[prompt_result]
        )

        # =====================================================================
        # Data Management Section
        # =====================================================================
        gr.Markdown("---")
        gr.Markdown("### Data Management")

        status = get_data_status()
        status_display = gr.Markdown(format_status_markdown(status))

        refresh_status_btn = gr.Button("Refresh Status", variant="secondary")

        def refresh_status():
            status = get_data_status()
            return format_status_markdown(status)

        refresh_status_btn.click(
            fn=refresh_status,
            outputs=[status_display]
        )

        # Reset controls
        gr.Markdown("---")
        gr.Markdown("### Reset Data")
        gr.Markdown("*Warning: These actions cannot be undone.*")

        confirm_checkbox = gr.Checkbox(
            label="I understand this action cannot be undone",
            value=False
        )

        with gr.Row():
            reset_extractions_btn = gr.Button(
                "Reset Extractions",
                variant="secondary",
                interactive=False
            )
            reset_all_btn = gr.Button(
                "Reset All Processed Data",
                variant="secondary",
                interactive=False
            )

        reset_result = gr.Markdown("")

        def toggle_reset_buttons(confirmed):
            return (
                gr.update(interactive=confirmed),
                gr.update(interactive=confirmed)
            )

        confirm_checkbox.change(
            fn=toggle_reset_buttons,
            inputs=[confirm_checkbox],
            outputs=[reset_extractions_btn, reset_all_btn]
        )

        def do_reset_extractions():
            result = reset_extractions()
            status = get_data_status()
            return (
                f"Deleted {result['deleted']} extraction files.",
                format_status_markdown(status),
                False,  # Uncheck the confirmation
                gr.update(interactive=False),  # Disable reset extractions button
                gr.update(interactive=False)   # Disable reset all button
            )

        def do_reset_all():
            result = reset_all_processed()
            status = get_data_status()
            msg_parts = [f"Deleted {result['extractions']} extraction files."]
            if result['consolidated']:
                msg_parts.append("Deleted consolidated.json.")
            if result['categorized']:
                msg_parts.append("Deleted categorized.json.")
            return (
                " ".join(msg_parts),
                format_status_markdown(status),
                False,  # Uncheck the confirmation
                gr.update(interactive=False),  # Disable reset extractions button
                gr.update(interactive=False)   # Disable reset all button
            )

        reset_extractions_btn.click(
            fn=do_reset_extractions,
            outputs=[reset_result, status_display, confirm_checkbox, reset_extractions_btn, reset_all_btn]
        )

        reset_all_btn.click(
            fn=do_reset_all,
            outputs=[reset_result, status_display, confirm_checkbox, reset_extractions_btn, reset_all_btn]
        )


# =============================================================================
# JAVASCRIPT FOR LIVE COLOR UPDATES
# =============================================================================

THEME_UPDATE_JS = """
function updateThemeColor(color) {
    // Generate shades from base color
    const hex = color.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);

    function shade(factor) {
        return '#' + [r, g, b].map(c =>
            Math.round(c * factor).toString(16).padStart(2, '0')
        ).join('');
    }

    const shades = {
        full: color,
        dark: shade(0.4),
        medium: shade(0.6),
        dim: shade(0.2),
        faint: shade(0.1)
    };

    // Update CSS variables
    document.documentElement.style.setProperty('--theme-color', shades.full);
    document.documentElement.style.setProperty('--theme-color-dark', shades.dark);
    document.documentElement.style.setProperty('--theme-color-medium', shades.medium);
    document.documentElement.style.setProperty('--theme-color-dim', shades.dim);
    document.documentElement.style.setProperty('--theme-color-faint', shades.faint);

    return color;
}
"""

# =============================================================================
# UPLOAD TAB
# =============================================================================

def create_upload_tab():
    """Create the JSON upload and auto-parse tab with support for multiple formats."""
    from config import load_config, save_config

    def handle_upload_and_parse(file_path, export_format: str):
        """Handle file upload and automatically trigger parsing.

        Args:
            file_path: Path to uploaded file
            export_format: 'chatgpt' or 'claude'
        """
        format_names = {'chatgpt': 'ChatGPT', 'claude': 'Claude'}
        format_name = format_names.get(export_format, export_format)

        if not file_path:
            yield f"No file uploaded. Drag and drop your {format_name} conversations.json file above.", None, gr.update(visible=False)
            return

        # Validate JSON structure
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                yield "Error: Invalid format. Expected an array of conversations.", None, gr.update(visible=False)
                return
            conversation_count = len(data)
        except json.JSONDecodeError as e:
            yield f"Error: Invalid JSON file.\n{e}", None, gr.update(visible=False)
            return
        except Exception as e:
            yield f"Error reading file: {e}", None, gr.update(visible=False)
            return

        # Target path for parser (format-specific)
        target_path = Path(f"data/{export_format}_conversations.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file if present
        log_lines = []
        if target_path.exists():
            backup_name = f"{export_format}_conversations.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = target_path.parent / backup_name
            try:
                shutil.copy2(target_path, backup_path)
                log_lines.append(f"Backed up existing file to {backup_name}")
            except Exception as e:
                log_lines.append(f"Warning: Could not create backup: {e}")

        # Copy uploaded file to target location
        try:
            shutil.copy2(file_path, target_path)
            log_lines.append(f"Uploaded {format_name} file with {conversation_count:,} conversations")
            log_lines.append("Starting parser...")
            log_lines.append("")
            yield "\n".join(log_lines), None, gr.update(visible=False)
        except Exception as e:
            yield f"Error copying file: {e}", None, gr.update(visible=False)
            return

        # Execute parser as subprocess with format argument
        try:
            process = subprocess.Popen(
                [sys.executable, "parser.py", "--format", export_format, "--input", str(target_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent,
                text=True,
                bufsize=1
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    log_lines.append(output)
                    yield "\n".join(log_lines), None, gr.update(visible=False)

            # Get final return code
            returncode = process.poll()
            stderr = process.stderr.read()

            if returncode == 0:
                # Parse successful - load manifest
                manifest_path = Path("data/parsed/_manifest.json")
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    stats = {
                        "total_conversations": manifest.get("total_conversations"),
                        "parsed": manifest.get("parsed"),
                        "skipped_trivial": manifest.get("skipped_trivial"),
                        "skipped_malformed": manifest.get("skipped_malformed"),
                        "date_range": manifest.get("date_range")
                    }

                    log_lines.append("")
                    log_lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Parsing complete!")
                    log_lines.append("")
                    log_lines.append("You can now go to the Extraction tab to extract insights from your conversations.")
                    yield "\n".join(log_lines), stats, gr.update(visible=True)
                else:
                    log_lines.append("\nWarning: Manifest file not found.")
                    yield "\n".join(log_lines), None, gr.update(visible=False)
            else:
                log_lines.append(f"\nError: Parser failed with code {returncode}")
                if stderr:
                    log_lines.append(f"Error details:\n{stderr}")
                yield "\n".join(log_lines), None, gr.update(visible=False)

        except Exception as e:
            log_lines.append(f"\nError running parser: {e}")
            yield "\n".join(log_lines), None, gr.update(visible=False)

    def handle_chatgpt_upload(file_path):
        """Handler for ChatGPT uploads."""
        yield from handle_upload_and_parse(file_path, 'chatgpt')

    def handle_claude_upload(file_path):
        """Handler for Claude uploads."""
        yield from handle_upload_and_parse(file_path, 'claude')

    with gr.Tab("Upload"):
        gr.Markdown("## Upload Conversation Export")
        gr.Markdown("""
Upload your conversation export file to get started. Select the tab matching your AI assistant.
Both ChatGPT and Claude exports are named `conversations.json` - use the appropriate tab for your source.
""")

        # Parsing filter settings
        with gr.Row():
            config = load_config()
            min_turns_input = gr.Number(
                label="Minimum conversation turns",
                value=config.get('min_turn_threshold', 4),
                minimum=1,
                maximum=100,
                step=1,
                info="Conversations with fewer turns are skipped during parsing"
            )
            save_threshold_btn = gr.Button("Save", size="sm", variant="secondary")
            threshold_status = gr.Markdown("")

        def save_min_turns_threshold(min_turns):
            """Save the minimum turns threshold to config."""
            try:
                current_config = load_config()
                current_config['min_turn_threshold'] = int(min_turns)
                save_config(current_config)
                return f"Saved: {int(min_turns)} turns"
            except Exception as e:
                return f"Error: {e}"

        save_threshold_btn.click(
            fn=save_min_turns_threshold,
            inputs=[min_turns_input],
            outputs=[threshold_status]
        )

        with gr.Tabs():
            # ChatGPT Upload Tab
            with gr.Tab("ChatGPT"):
                gr.Markdown("""
**How to export from ChatGPT:**
1. Go to [chat.openai.com](https://chat.openai.com)
2. Settings → Data Controls → Export data
3. Wait for the email with your download link
4. Download and extract the ZIP file
5. Drag the `conversations.json` file below
""")
                chatgpt_upload = gr.File(
                    label="Drop your ChatGPT conversations.json here",
                    file_types=[".json"],
                    file_count="single",
                    type="filepath"
                )

                chatgpt_status = gr.Textbox(
                    label="Parser Status",
                    value="Upload a ChatGPT export to begin...",
                    lines=15,
                    max_lines=25,
                    interactive=False
                )

                chatgpt_stats = gr.JSON(
                    label="Parse Statistics",
                    visible=False
                )

                chatgpt_upload.upload(
                    fn=handle_chatgpt_upload,
                    inputs=[chatgpt_upload],
                    outputs=[chatgpt_status, chatgpt_stats, chatgpt_stats]
                )

            # Claude Upload Tab
            with gr.Tab("Claude"):
                gr.Markdown("""
**How to export from Claude:**
1. Go to [claude.ai](https://claude.ai)
2. Click your profile icon → Settings
3. Account → Export Data
4. Wait for the email with your download link
5. Download and extract the ZIP file
6. Drag the `conversations.json` file below
""")
                claude_upload = gr.File(
                    label="Drop your Claude conversations.json here",
                    file_types=[".json"],
                    file_count="single",
                    type="filepath"
                )

                claude_status = gr.Textbox(
                    label="Parser Status",
                    value="Upload a Claude export to begin...",
                    lines=15,
                    max_lines=25,
                    interactive=False
                )

                claude_stats = gr.JSON(
                    label="Parse Statistics",
                    visible=False
                )

                claude_upload.upload(
                    fn=handle_claude_upload,
                    inputs=[claude_upload],
                    outputs=[claude_status, claude_stats, claude_stats]
                )


# =============================================================================
# EXPORT TAB
# =============================================================================

def create_export_tab():
    """Create the Obsidian export tab."""
    from pathlib import Path

    with gr.Tab("Export"):
        gr.Markdown("## Export to Obsidian")
        gr.Markdown("""
Export your Resurface data to an Obsidian-compatible markdown vault.

The export creates:
- **Conversations** - Full transcripts with metadata and wiki-links
- **Ideas** - Consolidated ideas organized by category (Quick Wins, Validate, Revive, Someday)
- **Problems & Workflows** - Pain points and processes you've explored
- **Tools** - Hub pages for each tool with backlinks to related content
- **Themes** - Theme pages from your passion profile
- **Maps of Content** - Navigation pages for exploring your knowledge base

Open the exported vault in Obsidian to visualize your brain map in the Graph View!
""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Export Options")
                include_conversations = gr.Checkbox(
                    label="Include full conversations",
                    value=True,
                    info="Export complete conversation transcripts"
                )
                include_consolidated = gr.Checkbox(
                    label="Include consolidated insights",
                    value=True,
                    info="Export merged ideas, problems, and workflows"
                )
                include_synthesized = gr.Checkbox(
                    label="Include synthesis outputs",
                    value=True,
                    info="Export passion profile and generated ideas"
                )
                clean_export = gr.Checkbox(
                    label="Clean export (delete existing vault)",
                    value=False,
                    info="Remove previous export before generating new one"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Graph View Tips")
                gr.Markdown("""
After opening in Obsidian:
1. Press `Ctrl/Cmd + G` for Graph View
2. Use filters to focus:
   - `-#type/conversation` hides conversations
   - `#category/quick_win` shows quick wins
   - `#emotion/excited` shows exciting topics
3. Color nodes by tag for visual grouping
4. Adjust depth slider to see connections
""")

        with gr.Row():
            export_btn = gr.Button("Export to Obsidian", variant="primary", size="lg")
            incremental_btn = gr.Button("Incremental Export", variant="secondary")

        export_progress = gr.Textbox(
            label="Export Progress",
            value="Click 'Export to Obsidian' to begin...",
            lines=8,
            interactive=False
        )

        with gr.Row():
            with gr.Column():
                export_stats = gr.JSON(
                    label="Export Statistics",
                    visible=False
                )
            with gr.Column():
                vault_path_display = gr.Textbox(
                    label="Vault Location",
                    value=str(Path("data/obsidian-vault").absolute()),
                    interactive=False,
                    visible=False
                )

        def run_export_handler(include_conv, include_cons, include_synth, clean):
            """Handle export button click with streaming progress."""
            try:
                from obsidian_exporter import run_export, get_status
                import time

                # Start export in background would be ideal, but for simplicity
                # we'll run synchronously and stream status
                yield (
                    "Starting export...",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

                stats = run_export(
                    include_conversations=include_conv,
                    include_consolidated=include_cons,
                    include_synthesized=include_synth,
                    clean_export=clean
                )

                if stats:
                    vault_path = str(Path("data/obsidian-vault").absolute())
                    msg = f"""Export complete!

Exported:
- {stats.get('conversations', 0)} conversations
- {stats.get('ideas', 0)} ideas
- {stats.get('problems', 0)} problems
- {stats.get('workflows', 0)} workflows
- {stats.get('tools', 0)} tool pages
- {stats.get('themes', 0)} theme pages
- {stats.get('generated', 0)} generated ideas

Vault location: {vault_path}

To view your brain map:
1. Open Obsidian
2. Click "Open folder as vault"
3. Select: {vault_path}
4. Press Ctrl/Cmd + G for Graph View"""
                    yield (
                        msg,
                        gr.update(value=stats, visible=True),
                        gr.update(value=vault_path, visible=True)
                    )
                else:
                    yield (
                        "Export failed - check console for errors",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield (
                    f"Export error: {str(e)}",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        def run_incremental_handler():
            """Handle incremental export."""
            try:
                from obsidian_exporter import run_incremental_export

                yield (
                    "Checking for new items...",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

                stats = run_incremental_export()

                if stats.get("new_conversations", 0) == 0 and not stats.get("conversations"):
                    yield (
                        "No new items to export. Your vault is up to date!",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                else:
                    vault_path = str(Path("data/obsidian-vault").absolute())
                    new_count = stats.get("new_conversations", 0)
                    msg = f"""Incremental export complete!

Added {new_count} new conversations.

Total in vault:
- {stats.get('conversations', 0)} conversations
- {stats.get('ideas', 0)} ideas
- {stats.get('tools', 0)} tools

Vault location: {vault_path}"""
                    yield (
                        msg,
                        gr.update(value=stats, visible=True),
                        gr.update(value=vault_path, visible=True)
                    )

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield (
                    f"Export error: {str(e)}",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        export_btn.click(
            fn=run_export_handler,
            inputs=[include_conversations, include_consolidated, include_synthesized, clean_export],
            outputs=[export_progress, export_stats, vault_path_display]
        )

        incremental_btn.click(
            fn=run_incremental_handler,
            inputs=[],
            outputs=[export_progress, export_stats, vault_path_display]
        )


# =============================================================================
# MAIN APP
# =============================================================================

def create_app():
    """Create the main Gradio app."""
    from config import load_config
    config = load_config()
    theme_color = config.get('theme_color', '#00ff00')

    # Generate theme components (passed to launch())
    theme = create_matrix_theme(theme_color)
    css = generate_matrix_css(theme_color)

    with gr.Blocks(title="Resurface") as app:
        # Store for use in launch()
        app._matrix_theme = theme
        app._matrix_css = css
        app._matrix_js = THEME_UPDATE_JS

        gr.Markdown("# Resurface")

        with gr.Tabs(selected="dashboard"):
            # Create tabs in display order (Dashboard remains default startup via selected=)
            create_settings_tab()
            create_upload_tab()
            browser_tab, browser_load_fn, browser_outputs = create_browser_tab()
            create_extraction_tab()
            dashboard_tab, dashboard_load_fn, dashboard_outputs = create_dashboard_tab()
            ideas_tab, ideas_load_fn, ideas_outputs = create_ideas_tab()
            problems_tab, problems_load_fn, problems_outputs = create_problems_tab()
            tools_tab, tools_load_fn, tools_outputs = create_tools_tab()
            emotions_tab, emotions_load_fn, emotions_outputs = create_emotions_tab()
            consolidated_tab, consolidated_load_fn, consolidated_outputs = create_consolidated_tab()
            categories_tab, categories_load_fn, categories_outputs = create_categories_tab()
            synthesis_tab, synthesis_load_fn, synthesis_outputs = create_synthesis_tab()
            create_export_tab()

        # Auto-load dashboard on start
        app.load(fn=dashboard_load_fn, outputs=dashboard_outputs)

        # Auto-refresh data when tabs are selected
        dashboard_tab.select(fn=dashboard_load_fn, outputs=dashboard_outputs)
        browser_tab.select(fn=browser_load_fn, outputs=browser_outputs)
        ideas_tab.select(fn=ideas_load_fn, outputs=ideas_outputs)
        problems_tab.select(fn=problems_load_fn, outputs=problems_outputs)
        tools_tab.select(fn=tools_load_fn, outputs=tools_outputs)
        emotions_tab.select(fn=emotions_load_fn, outputs=emotions_outputs)
        consolidated_tab.select(fn=consolidated_load_fn, outputs=consolidated_outputs)
        categories_tab.select(fn=categories_load_fn, outputs=categories_outputs)
        synthesis_tab.select(fn=synthesis_load_fn, outputs=synthesis_outputs)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=False,
        allowed_paths=["assets"],
        theme=app._matrix_theme,
        css=app._matrix_css,
        js=app._matrix_js
    )
