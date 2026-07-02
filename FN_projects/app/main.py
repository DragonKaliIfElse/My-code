import sys
import os
from openpyxl.chart import area_chart

from qt_core import *
from gui.windows.main_window.ui_main_window import UI_MainWindow
from gui.theme import *
from xlsEditor import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Bate Estaca")

        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)

        # Sidebar toggle
        self.ui.toggle_button.clicked.connect(self.toggle_button)

        # Navigation
        self.ui.btn_1.clicked.connect(self.show_page_add_planilha)

        # Page flow
        self.ui.ui_pages.adicionar_planilha_button.clicked.connect(self.show_page_formulario)
        self.ui.ui_pages.salvar_button.clicked.connect(self.show_page_golpes)
        self.ui.ui_pages.finalizar_button.clicked.connect(self.add_planilha_action)
        self.ui.ui_pages.cancelar_button.clicked.connect(self.show_page_add_planilha)

        # Back buttons
        self.ui.ui_pages.back_to_home.clicked.connect(self.show_page_add_planilha)
        self.ui.ui_pages.back_to_form.clicked.connect(self.show_page_formulario)

        self.show()

        # Fade effect helper
        self._opacity_effects = {}

    # ── Navigation ────────────────────────────────────
    def reset_selection(self):
        for btn in self.ui.left_menu.findChildren(QPushButton):
            try:
                btn.set_active(False)
            except:
                pass

    def show_page_add_planilha(self):
        self.reset_selection()
        self._fade_to(self.ui.ui_pages.add_planilha)
        self.ui.btn_1.set_active(True)
        self.ui.set_breadcrumb("Início")

    def show_page_formulario(self):
        self.reset_selection()
        self._fade_to(self.ui.ui_pages.formulario)
        self.ui.btn_1.set_active(True)
        self.ui.set_breadcrumb("Início > Formulário")

    def show_page_golpes(self):
        self.reset_selection()
        self._fade_to(self.ui.ui_pages.golpes)
        self.ui.btn_1.set_active(True)
        self.ui.set_breadcrumb("Início > Formulário > Golpes")

    # ── Fade transition ───────────────────────────────
    def _fade_to(self, page):
        self.ui.pages.setCurrentWidget(page)
        effect = QGraphicsOpacityEffect(page)
        effect.setOpacity(0.0)
        page.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(200)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._opacity_effects[id(page)] = (effect, anim)

    # ── Validation ────────────────────────────────────
    def _validate_form(self):
        required = {
            "nome_aba": "Nome da aba",
            "cliente": "Cliente",
            "obra": "Obra",
            "estaca": "Estaca",
            "comp_cravado": "Comp. cravado",
            "peso_martelo": "Peso do martelo",
            "altura_queda": "Altura de queda",
        }
        valid = True
        for field_name, label in required.items():
            field = getattr(self.ui.ui_pages, field_name, None)
            if not field or not field.text().strip():
                if hasattr(field, "set_error"):
                    field.set_error(f"{label} é obrigatório")
                valid = False
            else:
                if hasattr(field, "clear_error"):
                    field.clear_error()
        return valid

    def _validate_home(self):
        nome = self.ui.ui_pages.nome_planilha
        if not nome.text().strip():
            nome.setStyleSheet(STYLE_INPUT_ERROR)
            QToolTip.showText(nome.mapToGlobal(nome.rect().bottomLeft()),
                              "Digite um nome para a planilha", nome)
            return False
        nome.setStyleSheet(STYLE_INPUT)
        return True

    # ── Generate Excel ────────────────────────────────
    def add_planilha_action(self):
        if not self._validate_home():
            return
        if not self._validate_form():
            return

        nome_planilha = self.ui.ui_pages.nome_planilha.text()
        nome_arquivo = nome_planilha + '.xlsx'
        nome_aba = self.ui.ui_pages.nome_aba.text()
        cliente = self.ui.ui_pages.cliente.text()
        obra = self.ui.ui_pages.obra.text()
        local = self.ui.ui_pages.local.text()
        nega = self.ui.ui_pages.nega.text()
        area = self.ui.ui_pages.area.text()
        be = self.ui.ui_pages.be.text()
        estaca = self.ui.ui_pages.estaca.text()
        secao = self.ui.ui_pages.secao.text()
        data_inicial = self.ui.ui_pages.data_inicial.text()
        data_final = self.ui.ui_pages.data_final.text()
        comp_cravado = self.ui.ui_pages.comp_cravado.text()
        peso_martelo = self.ui.ui_pages.peso_martelo.text()
        altura_queda = self.ui.ui_pages.altura_queda.text()
        golpes = self.ui.ui_pages.get_golpes()

        self.ui.show_loading(True)

        QTimer.singleShot(100, lambda: self._gerar(
            nome_arquivo, nome_aba, cliente, obra, local,
            nega, area, be, estaca, secao,
            data_inicial, data_final, comp_cravado,
            peso_martelo, altura_queda, golpes
        ))

    def _gerar(self, nome_arquivo, nome_aba, cliente, obra, local,
               nega, area, be, estaca, secao,
               data_inicial, data_final, comp_cravado,
               peso_martelo, altura_queda, golpes):
        try:
            gera_arquivo(
                nome_arquivo=nome_arquivo,
                nome_aba=nome_aba,
                cliente=cliente,
                obra=obra,
                local=local,
                nega=nega,
                area=area,
                be=be,
                estaca=estaca,
                secao=secao,
                data_inicial=data_inicial,
                data_final=data_final,
                comp_cravado=float(comp_cravado),
                peso_martelo=peso_martelo,
                altura_queda=float(altura_queda),
                golpes=golpes,
            )
            self.ui.show_loading(False)
            self.ui.set_breadcrumb("Início > Pronto!")

            QMessageBox.information(
                self, "Sucesso",
                f"Planilha '{nome_arquivo}' gerada com sucesso!"
            )
        except Exception as e:
            self.ui.show_loading(False)
            QMessageBox.critical(
                self, "Erro",
                f"Ocorreu um erro ao gerar a planilha:\n{str(e)}"
            )

    # ── Sidebar toggle ────────────────────────────────
    def toggle_button(self):
        menu_width = self.ui.left_menu.width()
        width = 50 if menu_width == 50 else 240
        self.animation = QPropertyAnimation(self.ui.left_menu, b"minimumWidth")
        self.animation.setStartValue(menu_width)
        self.animation.setEndValue(width)
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.InOutCirc)
        self.animation.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())
