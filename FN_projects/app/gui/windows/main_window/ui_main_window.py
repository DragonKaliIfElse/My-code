from qt_core import *
from gui.pages.ui_pages import Ui_StackedWidget
from gui.widgets.py_push_button import PyPushButton
from gui.theme import *

class UI_MainWindow(object):
    def setup_ui(self, parent):
        if not parent.objectName():
            parent.setObjectName("MainWindow")

        parent.resize(1200, 720)
        parent.setMinimumSize(960, 540)
        parent.setStyleSheet(f"font-family: '{FONT_FAMILY}';")

        # CENTRAL FRAME
        self.central_frame = QFrame()
        self.central_frame.setStyleSheet(f"background-color: {BG_PRIMARY};")

        self.main_layout = QHBoxLayout(self.central_frame)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── LEFT MENU ─────────────────────────────────
        self.left_menu = QFrame()
        self.left_menu.setStyleSheet(f"background-color: {BG_SIDEBAR};")
        self.left_menu.setMaximumWidth(50)
        self.left_menu.setMinimumWidth(50)

        self.left_menu_layout = QVBoxLayout(self.left_menu)
        self.left_menu_layout.setContentsMargins(0, 0, 0, 0)
        self.left_menu_layout.setSpacing(0)

        # Top frame
        self.left_menu_top_frame = QFrame()
        self.left_menu_top_frame.setMinimumHeight(40)
        self.left_menu_top_frame.setObjectName("left_menu_top_frame")

        self.left_menu_top_layout = QVBoxLayout(self.left_menu_top_frame)
        self.left_menu_top_layout.setContentsMargins(0, 0, 0, 0)
        self.left_menu_top_layout.setSpacing(0)

        self.toggle_button = PyPushButton(
            text="Ocultar menu",
            icon_path="icon_menu.svg",
            btn_color=BG_SIDEBAR,
            btn_hover="#2d2e3e",
            btn_pressed="#181920",
        )
        self.btn_1 = PyPushButton(
            text="Página inicial",
            is_active=True,
            icon_path="icon_home.svg",
            btn_color=BG_SIDEBAR,
            btn_hover="#2d2e3e",
            btn_pressed="#181920",
        )
        self.left_menu_top_layout.addWidget(self.toggle_button)
        self.left_menu_top_layout.addWidget(self.btn_1)

        self.left_menu_spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.left_menu_bottom_frame = QFrame()
        self.left_menu_bottom_frame.setMinimumHeight(40)
        self.left_menu_bottom_frame.setObjectName("left_menu_bottom_frame")

        self.left_menu_bottom_layout = QVBoxLayout(self.left_menu_bottom_frame)
        self.left_menu_bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.left_menu_bottom_layout.setSpacing(0)

        self.settings_btn = PyPushButton(
            text="Configurações",
            icon_path="icon_settings.svg",
            btn_color=BG_SIDEBAR,
            btn_hover="#2d2e3e",
            btn_pressed="#181920",
        )

        self.left_menu_bottom_layout.addWidget(self.settings_btn)

        self.left_menu_label_version = QLabel("v1.0.0")
        self.left_menu_label_version.setAlignment(Qt.AlignCenter)
        self.left_menu_label_version.setMinimumHeight(30)
        self.left_menu_label_version.setMaximumHeight(30)
        self.left_menu_label_version.setStyleSheet(f"color: {TEXT_MUTED};")

        self.left_menu_layout.addWidget(self.left_menu_top_frame)
        self.left_menu_layout.addItem(self.left_menu_spacer)
        self.left_menu_layout.addWidget(self.left_menu_bottom_frame)
        self.left_menu_layout.addWidget(self.left_menu_label_version)

        # ── CONTENT ───────────────────────────────────
        self.content = QFrame()
        self.content.setStyleSheet(f"background-color: {BG_PRIMARY};")

        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        # TOP BAR
        self.top_bar = QFrame()
        self.top_bar.setMinimumHeight(36)
        self.top_bar.setMaximumHeight(36)
        self.top_bar.setStyleSheet(f"background-color: {BG_TOPBAR};")

        self.top_bar_layout = QHBoxLayout(self.top_bar)
        self.top_bar_layout.setContentsMargins(16, 0, 16, 0)
        self.top_bar_layout.setSpacing(8)

        self.top_label_left = QLabel("FN Engenharia")
        self.top_label_left.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT};
                font-weight: 700;
                font-size: {FONT_SIZE_BODY}pt;
            }}
        """)

        self.top_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.top_label_right = QLabel("Início")
        self.top_label_right.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: {FONT_SIZE_SMALL}pt;
            }}
        """)

        self.top_bar_layout.addWidget(self.top_label_left)
        self.top_bar_layout.addItem(self.top_spacer)
        self.top_bar_layout.addWidget(self.top_label_right)

        # PAGES
        self.pages = QStackedWidget()
        self.pages.setStyleSheet(f"background-color: {BG_PRIMARY};")
        self.ui_pages = Ui_StackedWidget()
        self.ui_pages.setupUi(self.pages)
        self.pages.setCurrentWidget(self.ui_pages.add_planilha)

        # BOTTOM BAR
        self.bottom_bar = QFrame()
        self.bottom_bar.setMinimumHeight(32)
        self.bottom_bar.setMaximumHeight(32)
        self.bottom_bar.setStyleSheet(f"background-color: {BG_TOPBAR};")

        self.bottom_bar_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_bar_layout.setContentsMargins(16, 0, 16, 0)
        self.bottom_bar_layout.setSpacing(8)

        self.bottom_label_left = QLabel("Criado por: Eduardo Cruz")
        self.bottom_label_left.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FONT_SIZE_SMALL}pt;")

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(6)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setMinimumWidth(160)
        self.progress_bar.setMaximumWidth(240)
        self.progress_bar.setStyleSheet(STYLE_PROGRESS_BAR)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)

        self.bottom_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.bottom_label_right = QLabel("© 2026")
        self.bottom_label_right.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FONT_SIZE_SMALL}pt;")

        self.bottom_bar_layout.addWidget(self.bottom_label_left)
        self.bottom_bar_layout.addWidget(self.progress_bar)
        self.bottom_bar_layout.addItem(self.bottom_spacer)
        self.bottom_bar_layout.addWidget(self.bottom_label_right)

        # Add to content
        self.content_layout.addWidget(self.top_bar)
        self.content_layout.addWidget(self.pages)
        self.content_layout.addWidget(self.bottom_bar)

        # Assemble main layout
        self.main_layout.addWidget(self.left_menu)
        self.main_layout.addWidget(self.content)

        parent.setCentralWidget(self.central_frame)

    # ── Public helpers ───────────────────────────────
    def set_breadcrumb(self, text):
        self.top_label_right.setText(text)

    def show_loading(self, visible=True):
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
