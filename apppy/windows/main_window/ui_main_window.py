from qt_core import *
from widgets.py_push_button import PyPushButton 
from Pages.ui_pages import Ui_application_pages  

class UI_MainWindow(object):
    def setup_ui(self,parent):
        if not parent.objectName():
            parent.setObjectName('MainWindow')
        #tamanho padrão e mínimo da tela
        parent.resize(1200,720)
        parent.setMinimumSize(960,540)
        
        #inicializa o frame
        self.central_frame = QFrame()
        
        #layout principal dividindo os frames em dois
        self.main_layout = QHBoxLayout(self.central_frame)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)
        
        #frame do menu esquerdo
        self.left_menu = QFrame()
        self.left_menu.setStyleSheet("background-color: #44475a")
        self.left_menu.setMaximumWidth(50)
        self.left_menu.setMinimumWidth(50)
        
        #layout do menu esquerdo
        self.left_menu_layout = QVBoxLayout(self.left_menu)
        self.left_menu_layout.setContentsMargins(0,0,0,0)
        self.left_menu_layout.setSpacing(0)
        
        #frame da parte de cima do menu
        self.left_menu_top = QFrame()
        self.left_menu_top.setMinimumHeight(50)
        self.left_menu_top.setObjectName("left_menu_top")
        self.left_menu_top.setStyleSheet("#left_menu_top {background-color: red;}")
        
        #layout da parte de cima do menu
        self.left_menu_top_layout = QVBoxLayout(self.left_menu_top)
        self.left_menu_top_layout.setContentsMargins(0,0,0,0)
        self.left_menu_top_layout.setSpacing(0)
        
        #botões da parte de cima do menu
        self.toggle_button = PyPushButton(text ="Toggle")
        self.btn_1 = QPushButton("1")
        self.btn_2 = QPushButton("2")

        #adiciona os botões no layout da parte de cima do menu
        self.left_menu_top_layout.addWidget(self.toggle_button)
        self.left_menu_top_layout.addWidget(self.btn_1)
        self.left_menu_top_layout.addWidget(self.btn_2)

        #espaçador do meio do menu
        self.left_menu_spacer =  QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding)
        
        #frame da parte de baixo do menu
        self.left_menu_bottom =  QFrame()
        self.left_menu_bottom.setMinimumHeight(50)
        self.left_menu_bottom.setObjectName("left_menu_bottom")
        self.left_menu_bottom.setStyleSheet("#left_menu_bottom {background-color: red;}")
        
        #layout da parte de baixo do menu
        self.left_menu_bottom_layout = QVBoxLayout(self.left_menu_bottom)
        self.left_menu_bottom_layout.setContentsMargins(0,0,0,0)
        self.left_menu_bottom_layout.setSpacing(0)
        
        #botão da parte de baixo do menu
        self.toggle_button_bot = QPushButton("Toggle")

        #adiciona botão inferior no layout
        self.left_menu_bottom_layout.addWidget(self.toggle_button_bot)
        
        #legenda da versão do aplicativo na parte inferior do menu
        self.left_menu_version = QLabel("v1.0.0")
        self.left_menu_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_menu_version.setMinimumHeight(30)
        self.left_menu_version.setMaximumHeight(30)
        
        #adiciona os frames da parte de baixo do menu
        self.left_menu_layout.addWidget(self.left_menu_top)
        self.left_menu_layout.addItem(self.left_menu_spacer)
        self.left_menu_layout.addWidget(self.left_menu_bottom)
        self.left_menu_layout.addWidget(self.left_menu_version)

        #inicializa o frame do conteúdo
        self.content = QFrame()
        self.content.setStyleSheet("background-color: #282a36")
        
        #layuot do conteúdo
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0,0,0,0)
        self.content_layout.setSpacing(0)
        
        #frame da barra superior
        self.top_bar = QFrame()
        self.top_bar.setMinimumHeight(30)
        self.top_bar.setMaximumHeight(30)
        self.top_bar.setStyleSheet("background-color: #21232d; color: #6272a4")
        
        #layout da barra superior
        self.top_bar_layout = QHBoxLayout(self.top_bar)
        self.top_bar_layout.setContentsMargins(10,0,10,0)
        
        #legenda a esquerda da barra superior
        self.top_bar_left = QLabel("Área para cadastro de novo Evento/Projeto")
        
        #espaãmeto da barra superior
        self.top_bar_spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        
        #legenda da parte direita da barra superior
        self.top_bar_right = QLabel("| PÁGINA DE CADASTRO")
        self.top_bar_right.setStyleSheet("font: 700 9pt 'Segoe UI'")

        #adiciona frames no layout da barra superior
        self.top_bar_layout.addWidget(self.top_bar_left)
        self.top_bar_layout.addItem(self.top_bar_spacer)
        self.top_bar_layout.addWidget(self.top_bar_right)

        #carrega as paginas do ui_pages.py
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("font-size: 12pt; color: #f8f8f2")
        self.ui_pages = Ui_application_pages()
        self.ui_pages.setupUi(self.pages)
        self.pages.setCurrentWidget(self.ui_pages.page1)

        #frame da barra inferior
        self.bottom_bar = QFrame()
        self.bottom_bar.setMinimumHeight(30)
        self.bottom_bar.setMaximumHeight(30)
        self.bottom_bar.setStyleSheet("background-color: #21232d; color: #6272a4")
        
        #layout da barra inferior
        self.bottom_bar_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_bar_layout.setContentsMargins(10,0,10,0)
        
        #legenda a direita da barra inferior
        self.bottom_bar_left = QLabel("By INSOLIS DATA SOLUTIONS")
        
        #espaçamento da barra inferior
        self.bottom_bar_spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        
        #legenda a direita da barra inferior
        self.bottom_bar_right = QLabel("© 2025")
        
        #adiciona frames no layout da barra inferior
        self.bottom_bar_layout.addWidget(self.bottom_bar_left)
        self.bottom_bar_layout.addItem(self.bottom_bar_spacer)
        self.bottom_bar_layout.addWidget(self.bottom_bar_right)
        
        #adiciona frames no layout de conteúdo
        self.content_layout.addWidget(self.top_bar)
        self.content_layout.addWidget(self.pages)
        self.content_layout.addWidget(self.bottom_bar)
        
        #adiciona os frames da esquerda e da direita
        self.main_layout.addWidget(self.left_menu)
        self.main_layout.addWidget(self.content)
        
        #seta o frame central
        parent.setCentralWidget(self.central_frame)
        
