from qt_core import *
from gui.theme import *

class Card(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            background-color: {BG_SURFACE};
            border: none;
            border-radius: {BORDER_RADIUS};
        """)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(24, 20, 24, 20)
        self._layout.setSpacing(12)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(STYLE_LABEL_SECTION)
            title_label.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(title_label)

    def addWidget(self, widget, stretch=0, alignment=Qt.AlignmentFlag.AlignLeft):
        self._layout.addWidget(widget, stretch, alignment)

    def addLayout(self, layout):
        self._layout.addLayout(layout)

    def addSpacing(self, space):
        self._layout.addSpacing(space)
