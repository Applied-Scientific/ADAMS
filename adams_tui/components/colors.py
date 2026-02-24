"""
Shared colour constants (Catppuccin Mocha palette).

Import from here instead of defining locally in each component.
"""

# ── Core palette ─────────────────────────────────────────────────
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BLUE = "#89b4fa"
MAUVE = "#cba6f7"
TEAL = "#94e2d5"
PEACH = "#fab387"
TEXT = "#cdd6f4"
DIM = "#6c7086"
SURFACE = "#181825"

# Provider → colour mapping for model tag badges
PROVIDER_COLORS = {
    "OpenAI": GREEN,
    "Anthropic": PEACH,
    "Gemini": BLUE,
}
