from qt_core import *
from gui.theme import *
from gui.widgets.card import Card
from gui.widgets.form_field import FormField

class Ui_StackedWidget(object):
    def setupUi(self, StackedWidget):
        StackedWidget.setStyleSheet(f"background-color: {BG_PRIMARY};")

        # ═══════════════════════════════════════════════
        # PAGE 1 — Home / Adicionar Planilha
        # ═══════════════════════════════════════════════
        self.add_planilha = QWidget()
        self.add_planilha.setObjectName("add_planilha")

        outer = QVBoxLayout(self.add_planilha)
        outer.setContentsMargins(0, 0, 0, 0)

        outer.addStretch()

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)

        card = Card("\nCriar Nova Planilha\n")
        card.setMinimumWidth(326)
        card.setMaximumWidth(365)
        card.layout().setContentsMargins(19, 14, 19, 14)
        card.layout().setSpacing(6)

        card.layout().insertStretch(0, 1)
        card.layout().insertStretch(2, 1)

        desc = QLabel("\nInsira o nome da planilha para começar.\n")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: {FONT_SIZE_BODY}pt;
                font-family: "{FONT_FAMILY}";
                border: 1px solid {BORDER};
                border-radius: {BORDER_RADIUS};
                padding: 6px 10px;
            }}
        """)
        card.addWidget(desc)

        self.nome_planilha = QLineEdit()
        self.nome_planilha.setPlaceholderText("Ex: Estaca 01 - Bloco A")
        self.nome_planilha.setStyleSheet(STYLE_INPUT)
        self.nome_planilha.setMinimumHeight(31)
        card.addWidget(self.nome_planilha)

        self.adicionar_planilha_button = QPushButton("Adicionar Planilha")
        self.adicionar_planilha_button.setCursor(Qt.PointingHandCursor)
        self.adicionar_planilha_button.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.adicionar_planilha_button.setMinimumHeight(31)
        card.addWidget(self.adicionar_planilha_button)

        card.layout().addStretch(1)

        middle.addStretch()
        middle.addWidget(card)
        middle.addStretch()

        outer.addLayout(middle)

        outer.addStretch()

        StackedWidget.addWidget(self.add_planilha)

        # ═══════════════════════════════════════════════
        # PAGE 2 — Formulário
        # ═══════════════════════════════════════════════
        self.formulario = QWidget()
        self.formulario.setObjectName("formulario")

        form_layout = QVBoxLayout(self.formulario)
        form_layout.setContentsMargins(24, 16, 24, 16)
        form_layout.setSpacing(16)

        # ── Header ────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        self.back_to_home = QPushButton("⬅")
        self.back_to_home.setCursor(Qt.PointingHandCursor)
        self.back_to_home.setFixedSize(36, 36)
        self.back_to_home.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 2px solid {BORDER};
                border-radius: {BORDER_RADIUS};
                color: {TEXT_SECONDARY};
                font-size: 18pt;
            }}
            QPushButton:hover {{
                background-color: {BG_INPUT};
                border-color: {BORDER_FOCUS};
                color: {TEXT_PRIMARY};
            }}
        """)
        header_row.addWidget(self.back_to_home)

        header = QLabel("Formulário de Dados")
        header.setStyleSheet(STYLE_LABEL_SECTION)
        header.setAlignment(Qt.AlignCenter)
        header_row.addWidget(header, 1)
        form_layout.addLayout(header_row)

        # Scroll area to handle smaller screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QWidget#scroll_content {{ background: transparent; }}
        """)

        scroll_content = QWidget()
        scroll_content.setObjectName("scroll_content")
        scroll_layout = QHBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(16)

        # ── Left column card ──────────────────────────
        card_left = Card("Informações do Projeto")
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.nome_aba = FormField("Nome da aba:", "Ex: Relatório")
        left_layout.addWidget(self.nome_aba)

        self.cliente = FormField("Cliente:", "Nome do cliente")
        left_layout.addWidget(self.cliente)

        self.obra = FormField("Obra:", "Nome da obra")
        left_layout.addWidget(self.obra)

        self.local = FormField("Local:", "Endereço / localização")
        left_layout.addWidget(self.local)

        self.area = FormField("Área:", "Ex: 150 m²")
        left_layout.addWidget(self.area)

        self.nega = FormField("Nega:", "Valor da nega")
        left_layout.addWidget(self.nega)

        self.be = FormField("B.E:", "B.E")
        left_layout.addWidget(self.be)

        card_left.addLayout(left_layout)
        scroll_layout.addWidget(card_left)

        # ── Right column card ─────────────────────────
        card_right = Card("Parâmetros Técnicos")
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.estaca = FormField("Estaca:", "Nº da estaca")
        right_layout.addWidget(self.estaca)

        self.secao = FormField("Seção:", "Seção transversal")
        right_layout.addWidget(self.secao)

        self.data_inicial = FormField("Data inicial:", "DD/MM/AAAA")
        right_layout.addWidget(self.data_inicial)

        self.data_final = FormField("Data final:", "DD/MM/AAAA")
        right_layout.addWidget(self.data_final)

        self.comp_cravado = FormField("Comp. cravado:", "Ex: 12.5")
        right_layout.addWidget(self.comp_cravado)

        self.peso_martelo = FormField("Peso do martelo:", "Ex: 2000 kg")
        right_layout.addWidget(self.peso_martelo)

        self.altura_queda = FormField("Altura de queda:", "Ex: 0.5 m")
        right_layout.addWidget(self.altura_queda)

        card_right.addLayout(right_layout)
        scroll_layout.addWidget(card_right)

        scroll.setWidget(scroll_content)
        form_layout.addWidget(scroll, 1)

        # ── Action buttons ────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.cancelar_button = QPushButton("Cancelar")
        self.cancelar_button.setCursor(Qt.PointingHandCursor)
        self.cancelar_button.setStyleSheet(STYLE_BUTTON_DANGER)
        self.cancelar_button.setMinimumHeight(40)
        self.cancelar_button.setMinimumWidth(140)
        btn_row.addWidget(self.cancelar_button)

        self.salvar_button = QPushButton("Salvar e Continuar")
        self.salvar_button.setCursor(Qt.PointingHandCursor)
        self.salvar_button.setStyleSheet(STYLE_BUTTON_SUCCESS)
        self.salvar_button.setMinimumHeight(40)
        self.salvar_button.setMinimumWidth(180)
        btn_row.addWidget(self.salvar_button)

        btn_row.addStretch()
        form_layout.addLayout(btn_row)

        # Backward-compat: keep direct references to inner QLineEdit
        self._bind_fields()

        StackedWidget.addWidget(self.formulario)

        # ═══════════════════════════════════════════════
        # PAGE 3 — Golpes
        # ═══════════════════════════════════════════════
        self.golpes = QWidget()
        self.golpes.setObjectName("golpes")

        golpes_layout = QVBoxLayout(self.golpes)
        golpes_layout.setContentsMargins(24, 16, 24, 16)
        golpes_layout.setSpacing(16)

        golpe_header_row = QHBoxLayout()
        golpe_header_row.setSpacing(8)

        self.back_to_form = QPushButton("⬅")
        self.back_to_form.setCursor(Qt.PointingHandCursor)
        self.back_to_form.setFixedSize(36, 36)
        self.back_to_form.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 2px solid {BORDER};
                border-radius: {BORDER_RADIUS};
                color: {TEXT_SECONDARY};
                font-size: 18pt;
            }}
            QPushButton:hover {{
                background-color: {BG_INPUT};
                border-color: {BORDER_FOCUS};
                color: {TEXT_PRIMARY};
            }}
        """)
        golpe_header_row.addWidget(self.back_to_form)

        golpe_header = QLabel("Registro de Golpes")
        golpe_header.setAlignment(Qt.AlignCenter)
        golpe_header.setStyleSheet(STYLE_LABEL_SECTION)
        golpe_header_row.addWidget(golpe_header, 1)
        golpes_layout.addLayout(golpe_header_row)

        table_card = Card()
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(8)

        self.golpe_table = QTableWidget()
        self.golpe_table.setObjectName("golpe_table")
        self.golpe_table.setColumnCount(3)
        self.golpe_table.setHorizontalHeaderLabels(["Nº", "Golpe", "Média"])
        self.golpe_table.setRowCount(56)
        self.golpe_table.setStyleSheet(STYLE_TABLE)
        self.golpe_table.verticalHeader().setVisible(False)
        self.golpe_table.setColumnWidth(0, 60)
        self.golpe_table.setColumnWidth(1, 120)
        self.golpe_table.setColumnWidth(2, 120)
        self.golpe_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.golpe_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.golpe_table.setAlternatingRowColors(True)
        self.golpe_table.setStyleSheet(self.golpe_table.styleSheet() + f"""
            QTableWidget {{ alternate-background-color: {BG_INPUT}; }}
        """)

        for row in range(56):
            item_num = QTableWidgetItem(str(row + 1))
            item_num.setTextAlignment(Qt.AlignCenter)
            item_num.setFlags(item_num.flags() & ~Qt.ItemIsEditable)
            self.golpe_table.setItem(row, 0, item_num)

            self.golpe_table.setItem(row, 1, QTableWidgetItem(""))
            item_avg = QTableWidgetItem("")
            item_avg.setFlags(item_avg.flags() & ~Qt.ItemIsEditable)
            self.golpe_table.setItem(row, 2, item_avg)

        self.golpe_table.itemChanged.connect(self._atualizar_media)

        table_layout.addWidget(self.golpe_table)
        table_card.addLayout(table_layout)
        golpes_layout.addWidget(table_card, 1)

        # Finalizar button
        finalizar_row = QHBoxLayout()
        finalizar_row.addStretch()

        self.finalizar_button = QPushButton("Finalizar e Gerar Excel")
        self.finalizar_button.setCursor(Qt.PointingHandCursor)
        self.finalizar_button.setStyleSheet(STYLE_BUTTON_SUCCESS)
        self.finalizar_button.setMinimumHeight(44)
        self.finalizar_button.setMinimumWidth(220)
        finalizar_row.addWidget(self.finalizar_button)

        finalizar_row.addStretch()
        golpes_layout.addLayout(finalizar_row)

        StackedWidget.addWidget(self.golpes)

    # ── Helpers ──────────────────────────────────────
    def _bind_fields(self):
        """Keep direct QLineEdit references for backward compatibility."""
        for field_name in [
            "nome_aba", "cliente", "obra", "local", "area",
            "nega", "be", "estaca", "secao", "data_inicial",
            "data_final", "comp_cravado", "peso_martelo", "altura_queda"
        ]:
            widget = getattr(self, field_name)
            setattr(self, field_name, widget.input)

    def get_golpes(self):
        values = []
        for row in range(self.golpe_table.rowCount()):
            item = self.golpe_table.item(row, 1)
            if item and item.text().strip():
                try:
                    values.append(int(item.text()))
                except ValueError:
                    pass
        return values

    def _atualizar_media(self, item):
        if item.column() != 1:
            return
        running_sum = 0
        running_count = 0
        for r in range(self.golpe_table.rowCount()):
            avg_item = self.golpe_table.item(r, 2)
            if not avg_item:
                continue
            val_item = self.golpe_table.item(r, 1)
            if val_item and val_item.text().strip():
                try:
                    running_sum += int(val_item.text())
                    running_count += 1
                except ValueError:
                    pass
            if running_count > 0:
                avg_item.setText(f"{running_sum / running_count:.1f}")
            else:
                avg_item.setText("")
