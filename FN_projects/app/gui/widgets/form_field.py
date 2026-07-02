import os
from qt_core import *
from gui.theme import *

ICONS_DIR = os.path.join(os.path.abspath(os.getcwd()), "gui", "images", "icons")

class FormField(QWidget):
    def __init__(self, label="", placeholder="", has_button=False,
                 button_text="", button_icon="icon_add.svg", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setStyleSheet(STYLE_LABEL_FIELD)
        layout.addWidget(self.label)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.setStyleSheet(STYLE_INPUT)
        self.input.setMinimumHeight(36)
        input_row.addWidget(self.input)

        self.button = None
        if has_button:
            self.button = QPushButton()
            self.button.setMinimumWidth(36)
            self.button.setMaximumWidth(36)
            self.button.setMinimumHeight(36)
            self.button.setMaximumHeight(36)
            self.button.setCursor(Qt.PointingHandCursor)
            icon_path = os.path.join(ICONS_DIR, button_icon)
            if os.path.exists(icon_path):
                self.button.setIcon(QIcon(icon_path))
                self.button.setIconSize(QSize(18, 18))
            else:
                self.button.setText(button_text or "+")
            self.button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT};
                    border: none;
                    border-radius: {BORDER_RADIUS};
                    color: {TEXT_PRIMARY};
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT_HOVER};
                }}
                QPushButton:pressed {{
                    background-color: {BG_SIDEBAR};
                }}
            """)
            input_row.addWidget(self.button)

        layout.addLayout(input_row)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"""
            QLabel {{
                color: {DANGER};
                font-size: 9pt;
                padding-left: 2px;
            }}
        """)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def text(self):
        return self.input.text()

    def setText(self, text):
        self.input.setText(text)

    def set_error(self, msg=""):
        if msg:
            self.input.setStyleSheet(STYLE_INPUT_ERROR)
            self.error_label.setText(msg)
            self.error_label.setVisible(True)
        else:
            self.input.setStyleSheet(STYLE_INPUT)
            self.error_label.setVisible(False)

    def clear_error(self):
        self.set_error()
