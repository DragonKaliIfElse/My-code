# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# IMPORT MODULES
from os.path import basename
import sys
import os

from openpyxl.chart import area_chart 

# IMPORT QT CORE
from qt_core import *

# IMPORT MAIN WINDOW
from gui.windows.main_window.ui_main_window import UI_MainWindow
from xlsEditor import *

# MAIN WINDOW
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Excel Bate Estaca")

        # SETUP MAIN WINDOW
        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)

        # Toggle button
        self.ui.toggle_button.clicked.connect(self.toggle_button)

        # Btn home
        self.ui.btn_1.clicked.connect(self.show_page_add_planilha)

        # Btn widgets
        #self.ui.btn_2.clicked.connect(self.show_page_formulario)

        # Btn settings
        #self.ui.settings_btn.clicked.connect(self.show_page_3)

        #função
        self.ui.ui_pages.adicionar_planilha_button.clicked.connect(self.show_page_formulario)
        self.ui.ui_pages.salvar_button.clicked.connect(self.show_page_golpes)
        self.ui.ui_pages.finalizar_button.clicked.connect(self.add_planilha_action)

        # EXIBI A NOSSA APLICAÇÃO
        self.show()

    # Change text - Home Page
    # Reset BTN Selection
    def reset_selection(self):
        for btn in self.ui.left_menu.findChildren(QPushButton):
            try:
                btn.set_active(False)
            except:
                pass
    
    # função de adicionar planilha 
    def add_planilha_action(self):
        nome_planilha = self.ui.ui_pages.nome_planilha.text()
        nome_arquivo = nome_planilha + '.xlsx'
        altura_queda = self.ui.ui_pages.altura_queda.text()
        comp_cravado = self.ui.ui_pages.comp_cravado.text()
        data_final = self.ui.ui_pages.data_final.text()
        data_inicial = self.ui.ui_pages.data_inicial.text()
        peso_martelo = self.ui.ui_pages.peso_martelo.text()
        secao = self.ui.ui_pages.secao.text()
        area = self.ui.ui_pages.area.text()
        be = self.ui.ui_pages.be.text()
        cliente = self.ui.ui_pages.cliente.text()
        estaca = self.ui.ui_pages.estaca.text()
        local = self.ui.ui_pages.local.text()
        nega = self.ui.ui_pages.nega.text()
        obra = self.ui.ui_pages.obra.text()
        nome_aba = self.ui.ui_pages.nome_aba.text()
        golpes = []

        for i in range(1, 57):
            campo = getattr(self.ui.ui_pages, f"golpe_{i}")
            golpe = campo.text()
            if golpe == '': continue;
            golpes.append(int(golpe))

        gera_arquivo(nome_arquivo=nome_arquivo,
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
                     golpes=golpes)

    # Btn home function
    def show_page_golpes(self):
        self.reset_selection()
        self.ui.pages.setCurrentWidget(self.ui.ui_pages.golpes)
        self.ui.btn_2.set_active(True)

    # Btn widgets function
    def show_page_formulario(self):
        self.reset_selection()
        self.ui.pages.setCurrentWidget(self.ui.ui_pages.formulario)
        self.ui.btn_2.set_active(True)

    def show_page_add_planilha(self):
        self.reset_selection()
        self.ui.pages.setCurrentWidget(self.ui.ui_pages.add_planilha)
        self.ui.btn_1.set_active(True)

    # Btn pase gettings
    def show_page_3(self):
        self.reset_selection()
        self.ui.pages.setCurrentWidget(self.ui.ui_pages.page_3)
        self.ui.settings_btn.set_active(True)

    # Toggle button
    def toggle_button(self):
        # Get menu width
        menu_width = self.ui.left_menu.width()

        # Check with
        width = 50
        if menu_width == 50:
            width = 240

        # Start animation
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
