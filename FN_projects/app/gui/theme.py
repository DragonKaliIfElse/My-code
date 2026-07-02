from PySide6.QtGui import QColor

# ── Palette ──────────────────────────────────────────
BG_PRIMARY       = "#1e1f2b"
BG_SURFACE       = "#282a36"
BG_SIDEBAR       = "#242531"
BG_INPUT         = "#2d2e3e"
BG_TOPBAR        = "#181920"
BORDER           = "#44475a"
BORDER_FOCUS     = "#6272a4"
BORDER_ERROR     = "#ff5555"
TEXT_PRIMARY     = "#f8f8f2"
TEXT_SECONDARY   = "#c3ccdf"
TEXT_MUTED       = "#6c7096"
ACCENT           = "#6272a4"
ACCENT_HOVER     = "#7a8bbf"
ACCENT_LIGHT     = "#8be9fd"
PRIMARY          = "#50fa7b"
PRIMARY_HOVER    = "#69ff94"
DANGER           = "#ff5555"
DANGER_HOVER     = "#ff7777"
WARNING          = "#ffb86c"

# ── Fonts ────────────────────────────────────────────
FONT_FAMILY = "Segoe UI"
FONT_SIZE_BODY = 11
FONT_SIZE_SMALL = 9
FONT_SIZE_TITLE = 26
FONT_SIZE_SECTION = 14
FONT_SIZE_INPUT = 11

# ── Border radius ──────────────────────────────────────
BORDER_RADIUS = "10px"

# ── Style strings ────────────────────────────────────

STYLE_INPUT = f"""
QLineEdit {{
    background-color: {BG_INPUT};
    padding: 8px 12px;
    border: 2px solid {BORDER};
    color: {TEXT_PRIMARY};
    border-radius: {BORDER_RADIUS};
    font-size: {FONT_SIZE_INPUT}pt;
    font-family: "{FONT_FAMILY}";
}}
QLineEdit:focus {{
    border: 2px solid {BORDER_FOCUS};
}}
QLineEdit:disabled {{
    opacity: 0.5;
}}
"""

STYLE_INPUT_ERROR = f"""
QLineEdit {{
    background-color: {BG_INPUT};
    padding: 8px 12px;
    border: 2px solid {BORDER_ERROR};
    color: {TEXT_PRIMARY};
    border-radius: {BORDER_RADIUS};
    font-size: {FONT_SIZE_INPUT}pt;
    font-family: "{FONT_FAMILY}";
}}
"""

STYLE_BUTTON_PRIMARY = f"""
QPushButton {{
    background-color: {ACCENT};
    padding: 8px 20px;
    border: none;
    color: {TEXT_PRIMARY};
    border-radius: {BORDER_RADIUS};
    font-weight: 600;
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
}}
QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton:pressed {{
    background-color: {BG_PRIMARY};
}}
"""

STYLE_BUTTON_SUCCESS = f"""
QPushButton {{
    background-color: {PRIMARY};
    padding: 8px 20px;
    border: none;
    color: {BG_PRIMARY};
    border-radius: {BORDER_RADIUS};
    font-weight: 700;
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
}}
QPushButton:hover {{
    background-color: {PRIMARY_HOVER};
}}
QPushButton:pressed {{
    background-color: {ACCENT_LIGHT};
}}
"""

STYLE_BUTTON_DANGER = f"""
QPushButton {{
    background-color: transparent;
    padding: 8px 20px;
    border: 2px solid {DANGER};
    color: {DANGER};
    border-radius: {BORDER_RADIUS};
    font-weight: 600;
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
}}
QPushButton:hover {{
    background-color: {DANGER};
    color: {TEXT_PRIMARY};
}}
QPushButton:pressed {{
    background-color: {DANGER_HOVER};
}}
"""

STYLE_BUTTON_GHOST = f"""
QPushButton {{
    background-color: transparent;
    padding: 8px 12px;
    border: 2px solid {BORDER};
    color: {TEXT_SECONDARY};
    border-radius: {BORDER_RADIUS};
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
}}
QPushButton:hover {{
    background-color: {BG_INPUT};
    border-color: {BORDER_FOCUS};
    color: {TEXT_PRIMARY};
}}
QPushButton:pressed {{
    background-color: {BORDER};
}}
"""

STYLE_LABEL = f"""
QLabel {{
    color: {TEXT_PRIMARY};
    font-family: "{FONT_FAMILY}";
}}
"""

STYLE_LABEL_SECTION = f"""
QLabel {{
    color: {ACCENT_LIGHT};
    font-size: {FONT_SIZE_SECTION}pt;
    font-weight: 700;
    font-family: "{FONT_FAMILY}";
    padding-bottom: 4px;
}}
"""

STYLE_LABEL_FIELD = f"""
QLabel {{
    color: {TEXT_SECONDARY};
    font-size: {FONT_SIZE_BODY}pt;
    font-weight: 600;
    font-family: "{FONT_FAMILY}";
    padding-left: 2px;
}}
"""

STYLE_TABLE = f"""
QTableWidget {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {BORDER_RADIUS};
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
    gridline-color: {BORDER};
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {BG_SIDEBAR};
    color: {TEXT_SECONDARY};
    padding: 6px;
    border: none;
    font-weight: 700;
    font-size: {FONT_SIZE_BODY}pt;
    font-family: "{FONT_FAMILY}";
}}
"""

STYLE_SCROLLBAR = f"""
QScrollBar:vertical {{
    background-color: {BG_SURFACE};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: {BG_SURFACE};
    height: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""

STYLE_PROGRESS_BAR = f"""
QProgressBar {{
    background-color: {BG_INPUT};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
"""
