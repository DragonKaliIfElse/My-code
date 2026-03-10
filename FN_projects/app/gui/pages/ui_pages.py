# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pagesVHTzfL.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QStackedWidget, QVBoxLayout, QWidget)

class Ui_StackedWidget(object):
    def setupUi(self, StackedWidget):
        if not StackedWidget.objectName():
            StackedWidget.setObjectName(u"StackedWidget")
        StackedWidget.resize(918, 558)
        self.add_planilha = QWidget()
        self.add_planilha.setObjectName(u"add_planilha")
        self.horizontalLayout = QHBoxLayout(self.add_planilha)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.frame = QFrame(self.add_planilha)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(500, 70))
        self.frame.setMaximumSize(QSize(500, 70))
        self.frame.setStyleSheet(u"border: 0px;")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayoutWidget = QWidget(self.frame)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 501, 71))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 30))
        self.label.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"font: 600 11pt \"URW Gothic\";")

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.nome_planilha = QLineEdit(self.verticalLayoutWidget)
        self.nome_planilha.setObjectName(u"nome_planilha")
        self.nome_planilha.setMaximumSize(QSize(16777215, 35))
        self.nome_planilha.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.horizontalLayout_3.addWidget(self.nome_planilha)

        self.adicionar_planilha_button = QPushButton(self.verticalLayoutWidget)
        self.adicionar_planilha_button.setObjectName(u"adicionar_planilha_button")
        self.adicionar_planilha_button.setMinimumSize(QSize(115, 0))
        self.adicionar_planilha_button.setMaximumSize(QSize(115, 16777215))
        self.adicionar_planilha_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.horizontalLayout_3.addWidget(self.adicionar_planilha_button, 0, Qt.AlignmentFlag.AlignRight)


        self.verticalLayout.addLayout(self.horizontalLayout_3)


        self.horizontalLayout.addWidget(self.frame, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        StackedWidget.addWidget(self.add_planilha)
        self.formulario = QWidget()
        self.formulario.setObjectName(u"formulario")
        self.verticalLayout_2 = QVBoxLayout(self.formulario)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.frame_2 = QFrame(self.formulario)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(900, 540))
        self.frame_2.setStyleSheet(u"border: 0px;")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayoutWidget_2 = QWidget(self.frame_2)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(-1, -1, 901, 541))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.verticalLayoutWidget_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setStyleSheet(u"font: 600 26pt \"URW Gothic\";\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout_3.addWidget(self.label_2, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.label_3 = QLabel(self.verticalLayoutWidget_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_3)

        self.label_5 = QLabel(self.verticalLayoutWidget_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_5)

        self.label_8 = QLabel(self.verticalLayoutWidget_2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_8)

        self.label_10 = QLabel(self.verticalLayoutWidget_2)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_10)

        self.label_17 = QLabel(self.verticalLayoutWidget_2)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_17)

        self.label_11 = QLabel(self.verticalLayoutWidget_2)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_11)

        self.label_9 = QLabel(self.verticalLayoutWidget_2)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_7.addWidget(self.label_9)


        self.horizontalLayout_14.addLayout(self.verticalLayout_7)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.nome_aba = QLineEdit(self.verticalLayoutWidget_2)
        self.nome_aba.setObjectName(u"nome_aba")
        self.nome_aba.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.nome_aba)

        self.cliente = QLineEdit(self.verticalLayoutWidget_2)
        self.cliente.setObjectName(u"cliente")
        self.cliente.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.cliente)

        self.obra = QLineEdit(self.verticalLayoutWidget_2)
        self.obra.setObjectName(u"obra")
        self.obra.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.obra)

        self.local = QLineEdit(self.verticalLayoutWidget_2)
        self.local.setObjectName(u"local")
        self.local.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.local)

        self.area = QLineEdit(self.verticalLayoutWidget_2)
        self.area.setObjectName(u"area")
        self.area.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.area)

        self.nega = QLineEdit(self.verticalLayoutWidget_2)
        self.nega.setObjectName(u"nega")
        self.nega.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.nega)

        self.be = QLineEdit(self.verticalLayoutWidget_2)
        self.be.setObjectName(u"be")
        self.be.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_6.addWidget(self.be)


        self.horizontalLayout_14.addLayout(self.verticalLayout_6)

        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.cliente_button = QPushButton(self.verticalLayoutWidget_2)
        self.cliente_button.setObjectName(u"cliente_button")
        self.cliente_button.setMinimumSize(QSize(70, 0))
        self.cliente_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.cliente_button)

        self.obra_button = QPushButton(self.verticalLayoutWidget_2)
        self.obra_button.setObjectName(u"obra_button")
        self.obra_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.obra_button)

        self.local_button = QPushButton(self.verticalLayoutWidget_2)
        self.local_button.setObjectName(u"local_button")
        self.local_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.local_button)

        self.nega_button = QPushButton(self.verticalLayoutWidget_2)
        self.nega_button.setObjectName(u"nega_button")
        self.nega_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.nega_button)

        self.be_button = QPushButton(self.verticalLayoutWidget_2)
        self.be_button.setObjectName(u"be_button")
        self.be_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.be_button)

        self.estaca_button = QPushButton(self.verticalLayoutWidget_2)
        self.estaca_button.setObjectName(u"estaca_button")
        self.estaca_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.estaca_button)

        self.estaca_button_2 = QPushButton(self.verticalLayoutWidget_2)
        self.estaca_button_2.setObjectName(u"estaca_button_2")
        self.estaca_button_2.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_11.addWidget(self.estaca_button_2)


        self.horizontalLayout_14.addLayout(self.verticalLayout_11)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_7 = QLabel(self.verticalLayoutWidget_2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_7)

        self.label_6 = QLabel(self.verticalLayoutWidget_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_6)

        self.label_12 = QLabel(self.verticalLayoutWidget_2)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_12)

        self.label_13 = QLabel(self.verticalLayoutWidget_2)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_13)

        self.label_14 = QLabel(self.verticalLayoutWidget_2)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_14)

        self.label_15 = QLabel(self.verticalLayoutWidget_2)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_15)

        self.label_16 = QLabel(self.verticalLayoutWidget_2)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 255, 255);\n"
"	font: 600 12pt \"URW Gothic\";\n"
"}")

        self.verticalLayout_8.addWidget(self.label_16)


        self.horizontalLayout_14.addLayout(self.verticalLayout_8)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.estaca = QLineEdit(self.verticalLayoutWidget_2)
        self.estaca.setObjectName(u"estaca")
        self.estaca.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.estaca)

        self.secao = QLineEdit(self.verticalLayoutWidget_2)
        self.secao.setObjectName(u"secao")
        self.secao.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.secao)

        self.data_inicial = QLineEdit(self.verticalLayoutWidget_2)
        self.data_inicial.setObjectName(u"data_inicial")
        self.data_inicial.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.data_inicial)

        self.data_final = QLineEdit(self.verticalLayoutWidget_2)
        self.data_final.setObjectName(u"data_final")
        self.data_final.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.data_final)

        self.comp_cravado = QLineEdit(self.verticalLayoutWidget_2)
        self.comp_cravado.setObjectName(u"comp_cravado")
        self.comp_cravado.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.comp_cravado)

        self.peso_martelo = QLineEdit(self.verticalLayoutWidget_2)
        self.peso_martelo.setObjectName(u"peso_martelo")
        self.peso_martelo.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.peso_martelo)

        self.altura_queda = QLineEdit(self.verticalLayoutWidget_2)
        self.altura_queda.setObjectName(u"altura_queda")
        self.altura_queda.setStyleSheet(u"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	padding: 8px;\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")

        self.verticalLayout_5.addWidget(self.altura_queda)


        self.horizontalLayout_14.addLayout(self.verticalLayout_5)

        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.secao_button = QPushButton(self.verticalLayoutWidget_2)
        self.secao_button.setObjectName(u"secao_button")
        self.secao_button.setMinimumSize(QSize(70, 0))
        self.secao_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.secao_button)

        self.data_inicial_button = QPushButton(self.verticalLayoutWidget_2)
        self.data_inicial_button.setObjectName(u"data_inicial_button")
        self.data_inicial_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.data_inicial_button)

        self.data_final_button = QPushButton(self.verticalLayoutWidget_2)
        self.data_final_button.setObjectName(u"data_final_button")
        self.data_final_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.data_final_button)

        self.comp_cravado_button = QPushButton(self.verticalLayoutWidget_2)
        self.comp_cravado_button.setObjectName(u"comp_cravado_button")
        self.comp_cravado_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.comp_cravado_button)

        self.peso_martelo_button = QPushButton(self.verticalLayoutWidget_2)
        self.peso_martelo_button.setObjectName(u"peso_martelo_button")
        self.peso_martelo_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.peso_martelo_button)

        self.altura_queda_button = QPushButton(self.verticalLayoutWidget_2)
        self.altura_queda_button.setObjectName(u"altura_queda_button")
        self.altura_queda_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.altura_queda_button)

        self.altura_queda_button_2 = QPushButton(self.verticalLayoutWidget_2)
        self.altura_queda_button_2.setObjectName(u"altura_queda_button_2")
        self.altura_queda_button_2.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.verticalLayout_12.addWidget(self.altura_queda_button_2)


        self.horizontalLayout_14.addLayout(self.verticalLayout_12)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_14)


        self.verticalLayout_3.addLayout(self.horizontalLayout_12)

        self.verticalSpacer = QSpacerItem(20, 100, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.salvar_button = QPushButton(self.verticalLayoutWidget_2)
        self.salvar_button.setObjectName(u"salvar_button")
        self.salvar_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.horizontalLayout_13.addWidget(self.salvar_button)

        self.cancelar_button = QPushButton(self.verticalLayoutWidget_2)
        self.cancelar_button.setObjectName(u"cancelar_button")
        self.cancelar_button.setStyleSheet(u"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}")

        self.horizontalLayout_13.addWidget(self.cancelar_button)


        self.verticalLayout_3.addLayout(self.horizontalLayout_13)


        self.verticalLayout_2.addWidget(self.frame_2, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        StackedWidget.addWidget(self.formulario)
        self.golpes = QWidget()
        self.golpes.setObjectName(u"golpes")
        self.golpes.setStyleSheet(u"QLabel{\n"
"	font: 600 12pt \"URW Gothic\";\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton{\n"
"	background-color: #44475a;\n"
"	padding: 8px;\n"
"	border: 1px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85,170,255)\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255,0,127)\n"
"}\n"
"QLineEdit {\n"
"	background-color: rgb(68,71,90);\n"
"	border:  2px solid #c3ccdf;\n"
"	color: rgb(255,255,255);\n"
"	border-radius: 10px;\n"
"}")
        self.horizontalLayout_4 = QHBoxLayout(self.golpes)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.frame_3 = QFrame(self.golpes)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayoutWidget_5 = QWidget(self.frame_3)
        self.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(3, 6, 891, 531))
        self.verticalLayout_10 = QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.verticalLayoutWidget_5)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setStyleSheet(u"QLabel{\n"
"	font: 600 26pt \"URW Gothic\";\n"
"	color: rgb(255, 255, 255);\n"
"}")

        self.verticalLayout_10.addWidget(self.label_4, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_19 = QVBoxLayout()
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.label_18 = QLabel(self.verticalLayoutWidget_5)
        self.label_18.setObjectName(u"label_18")

        self.verticalLayout_19.addWidget(self.label_18)

        self.label_27 = QLabel(self.verticalLayoutWidget_5)
        self.label_27.setObjectName(u"label_27")

        self.verticalLayout_19.addWidget(self.label_27)

        self.label_28 = QLabel(self.verticalLayoutWidget_5)
        self.label_28.setObjectName(u"label_28")

        self.verticalLayout_19.addWidget(self.label_28)

        self.label_30 = QLabel(self.verticalLayoutWidget_5)
        self.label_30.setObjectName(u"label_30")

        self.verticalLayout_19.addWidget(self.label_30)

        self.label_29 = QLabel(self.verticalLayoutWidget_5)
        self.label_29.setObjectName(u"label_29")

        self.verticalLayout_19.addWidget(self.label_29)

        self.label_26 = QLabel(self.verticalLayoutWidget_5)
        self.label_26.setObjectName(u"label_26")

        self.verticalLayout_19.addWidget(self.label_26)

        self.label_23 = QLabel(self.verticalLayoutWidget_5)
        self.label_23.setObjectName(u"label_23")

        self.verticalLayout_19.addWidget(self.label_23)

        self.label_24 = QLabel(self.verticalLayoutWidget_5)
        self.label_24.setObjectName(u"label_24")

        self.verticalLayout_19.addWidget(self.label_24)

        self.label_25 = QLabel(self.verticalLayoutWidget_5)
        self.label_25.setObjectName(u"label_25")

        self.verticalLayout_19.addWidget(self.label_25)

        self.label_22 = QLabel(self.verticalLayoutWidget_5)
        self.label_22.setObjectName(u"label_22")

        self.verticalLayout_19.addWidget(self.label_22)

        self.label_21 = QLabel(self.verticalLayoutWidget_5)
        self.label_21.setObjectName(u"label_21")

        self.verticalLayout_19.addWidget(self.label_21)

        self.label_20 = QLabel(self.verticalLayoutWidget_5)
        self.label_20.setObjectName(u"label_20")

        self.verticalLayout_19.addWidget(self.label_20)

        self.label_74 = QLabel(self.verticalLayoutWidget_5)
        self.label_74.setObjectName(u"label_74")

        self.verticalLayout_19.addWidget(self.label_74)

        self.label_19 = QLabel(self.verticalLayoutWidget_5)
        self.label_19.setObjectName(u"label_19")

        self.verticalLayout_19.addWidget(self.label_19)


        self.horizontalLayout_5.addLayout(self.verticalLayout_19)

        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.golpe_1 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_1.setObjectName(u"golpe_1")

        self.verticalLayout_14.addWidget(self.golpe_1)

        self.golpe_2 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_2.setObjectName(u"golpe_2")

        self.verticalLayout_14.addWidget(self.golpe_2)

        self.golpe_3 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_3.setObjectName(u"golpe_3")

        self.verticalLayout_14.addWidget(self.golpe_3)

        self.golpe_4 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_4.setObjectName(u"golpe_4")

        self.verticalLayout_14.addWidget(self.golpe_4)

        self.golpe_5 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_5.setObjectName(u"golpe_5")

        self.verticalLayout_14.addWidget(self.golpe_5)

        self.golpe_6 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_6.setObjectName(u"golpe_6")

        self.verticalLayout_14.addWidget(self.golpe_6)

        self.golpe_7 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_7.setObjectName(u"golpe_7")

        self.verticalLayout_14.addWidget(self.golpe_7)

        self.golpe_8 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_8.setObjectName(u"golpe_8")

        self.verticalLayout_14.addWidget(self.golpe_8)

        self.golpe_9 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_9.setObjectName(u"golpe_9")

        self.verticalLayout_14.addWidget(self.golpe_9)

        self.golpe_10 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_10.setObjectName(u"golpe_10")

        self.verticalLayout_14.addWidget(self.golpe_10)

        self.golpe_11 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_11.setObjectName(u"golpe_11")

        self.verticalLayout_14.addWidget(self.golpe_11)

        self.golpe_12 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_12.setObjectName(u"golpe_12")

        self.verticalLayout_14.addWidget(self.golpe_12)

        self.golpe_13 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_13.setObjectName(u"golpe_13")

        self.verticalLayout_14.addWidget(self.golpe_13)

        self.golpe_14 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_14.setObjectName(u"golpe_14")

        self.verticalLayout_14.addWidget(self.golpe_14)


        self.horizontalLayout_5.addLayout(self.verticalLayout_14)

        self.verticalLayout_20 = QVBoxLayout()
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.label_32 = QLabel(self.verticalLayoutWidget_5)
        self.label_32.setObjectName(u"label_32")

        self.verticalLayout_20.addWidget(self.label_32)

        self.label_33 = QLabel(self.verticalLayoutWidget_5)
        self.label_33.setObjectName(u"label_33")

        self.verticalLayout_20.addWidget(self.label_33)

        self.label_34 = QLabel(self.verticalLayoutWidget_5)
        self.label_34.setObjectName(u"label_34")

        self.verticalLayout_20.addWidget(self.label_34)

        self.label_35 = QLabel(self.verticalLayoutWidget_5)
        self.label_35.setObjectName(u"label_35")

        self.verticalLayout_20.addWidget(self.label_35)

        self.label_36 = QLabel(self.verticalLayoutWidget_5)
        self.label_36.setObjectName(u"label_36")

        self.verticalLayout_20.addWidget(self.label_36)

        self.label_37 = QLabel(self.verticalLayoutWidget_5)
        self.label_37.setObjectName(u"label_37")

        self.verticalLayout_20.addWidget(self.label_37)

        self.label_38 = QLabel(self.verticalLayoutWidget_5)
        self.label_38.setObjectName(u"label_38")

        self.verticalLayout_20.addWidget(self.label_38)

        self.label_39 = QLabel(self.verticalLayoutWidget_5)
        self.label_39.setObjectName(u"label_39")

        self.verticalLayout_20.addWidget(self.label_39)

        self.label_40 = QLabel(self.verticalLayoutWidget_5)
        self.label_40.setObjectName(u"label_40")

        self.verticalLayout_20.addWidget(self.label_40)

        self.label_41 = QLabel(self.verticalLayoutWidget_5)
        self.label_41.setObjectName(u"label_41")

        self.verticalLayout_20.addWidget(self.label_41)

        self.label_42 = QLabel(self.verticalLayoutWidget_5)
        self.label_42.setObjectName(u"label_42")

        self.verticalLayout_20.addWidget(self.label_42)

        self.label_43 = QLabel(self.verticalLayoutWidget_5)
        self.label_43.setObjectName(u"label_43")

        self.verticalLayout_20.addWidget(self.label_43)

        self.label_44 = QLabel(self.verticalLayoutWidget_5)
        self.label_44.setObjectName(u"label_44")

        self.verticalLayout_20.addWidget(self.label_44)

        self.label_45 = QLabel(self.verticalLayoutWidget_5)
        self.label_45.setObjectName(u"label_45")

        self.verticalLayout_20.addWidget(self.label_45)


        self.horizontalLayout_5.addLayout(self.verticalLayout_20)

        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.golpe_15 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_15.setObjectName(u"golpe_15")

        self.verticalLayout_16.addWidget(self.golpe_15)

        self.golpe_16 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_16.setObjectName(u"golpe_16")

        self.verticalLayout_16.addWidget(self.golpe_16)

        self.golpe_17 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_17.setObjectName(u"golpe_17")

        self.verticalLayout_16.addWidget(self.golpe_17)

        self.golpe_18 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_18.setObjectName(u"golpe_18")

        self.verticalLayout_16.addWidget(self.golpe_18)

        self.golpe_19 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_19.setObjectName(u"golpe_19")

        self.verticalLayout_16.addWidget(self.golpe_19)

        self.golpe_20 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_20.setObjectName(u"golpe_20")

        self.verticalLayout_16.addWidget(self.golpe_20)

        self.golpe_21 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_21.setObjectName(u"golpe_21")

        self.verticalLayout_16.addWidget(self.golpe_21)

        self.golpe_22 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_22.setObjectName(u"golpe_22")

        self.verticalLayout_16.addWidget(self.golpe_22)

        self.golpe_23 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_23.setObjectName(u"golpe_23")

        self.verticalLayout_16.addWidget(self.golpe_23)

        self.golpe_24 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_24.setObjectName(u"golpe_24")

        self.verticalLayout_16.addWidget(self.golpe_24)

        self.golpe_25 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_25.setObjectName(u"golpe_25")

        self.verticalLayout_16.addWidget(self.golpe_25)

        self.golpe_26 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_26.setObjectName(u"golpe_26")

        self.verticalLayout_16.addWidget(self.golpe_26)

        self.golpe_27 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_27.setObjectName(u"golpe_27")

        self.verticalLayout_16.addWidget(self.golpe_27)

        self.golpe_28 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_28.setObjectName(u"golpe_28")

        self.verticalLayout_16.addWidget(self.golpe_28)


        self.horizontalLayout_5.addLayout(self.verticalLayout_16)

        self.verticalLayout_21 = QVBoxLayout()
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.label_46 = QLabel(self.verticalLayoutWidget_5)
        self.label_46.setObjectName(u"label_46")

        self.verticalLayout_21.addWidget(self.label_46)

        self.label_47 = QLabel(self.verticalLayoutWidget_5)
        self.label_47.setObjectName(u"label_47")

        self.verticalLayout_21.addWidget(self.label_47)

        self.label_48 = QLabel(self.verticalLayoutWidget_5)
        self.label_48.setObjectName(u"label_48")

        self.verticalLayout_21.addWidget(self.label_48)

        self.label_49 = QLabel(self.verticalLayoutWidget_5)
        self.label_49.setObjectName(u"label_49")

        self.verticalLayout_21.addWidget(self.label_49)

        self.label_50 = QLabel(self.verticalLayoutWidget_5)
        self.label_50.setObjectName(u"label_50")

        self.verticalLayout_21.addWidget(self.label_50)

        self.label_51 = QLabel(self.verticalLayoutWidget_5)
        self.label_51.setObjectName(u"label_51")

        self.verticalLayout_21.addWidget(self.label_51)

        self.label_52 = QLabel(self.verticalLayoutWidget_5)
        self.label_52.setObjectName(u"label_52")

        self.verticalLayout_21.addWidget(self.label_52)

        self.label_53 = QLabel(self.verticalLayoutWidget_5)
        self.label_53.setObjectName(u"label_53")

        self.verticalLayout_21.addWidget(self.label_53)

        self.label_54 = QLabel(self.verticalLayoutWidget_5)
        self.label_54.setObjectName(u"label_54")

        self.verticalLayout_21.addWidget(self.label_54)

        self.label_55 = QLabel(self.verticalLayoutWidget_5)
        self.label_55.setObjectName(u"label_55")

        self.verticalLayout_21.addWidget(self.label_55)

        self.label_56 = QLabel(self.verticalLayoutWidget_5)
        self.label_56.setObjectName(u"label_56")

        self.verticalLayout_21.addWidget(self.label_56)

        self.label_57 = QLabel(self.verticalLayoutWidget_5)
        self.label_57.setObjectName(u"label_57")

        self.verticalLayout_21.addWidget(self.label_57)

        self.label_58 = QLabel(self.verticalLayoutWidget_5)
        self.label_58.setObjectName(u"label_58")

        self.verticalLayout_21.addWidget(self.label_58)

        self.label_59 = QLabel(self.verticalLayoutWidget_5)
        self.label_59.setObjectName(u"label_59")

        self.verticalLayout_21.addWidget(self.label_59)


        self.horizontalLayout_5.addLayout(self.verticalLayout_21)

        self.verticalLayout_17 = QVBoxLayout()
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.golpe_29 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_29.setObjectName(u"golpe_29")

        self.verticalLayout_17.addWidget(self.golpe_29)

        self.golpe_30 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_30.setObjectName(u"golpe_30")

        self.verticalLayout_17.addWidget(self.golpe_30)

        self.golpe_31 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_31.setObjectName(u"golpe_31")

        self.verticalLayout_17.addWidget(self.golpe_31)

        self.golpe_32 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_32.setObjectName(u"golpe_32")

        self.verticalLayout_17.addWidget(self.golpe_32)

        self.golpe_33 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_33.setObjectName(u"golpe_33")

        self.verticalLayout_17.addWidget(self.golpe_33)

        self.golpe_34 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_34.setObjectName(u"golpe_34")

        self.verticalLayout_17.addWidget(self.golpe_34)

        self.golpe_35 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_35.setObjectName(u"golpe_35")

        self.verticalLayout_17.addWidget(self.golpe_35)

        self.golpe_36 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_36.setObjectName(u"golpe_36")

        self.verticalLayout_17.addWidget(self.golpe_36)

        self.golpe_37 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_37.setObjectName(u"golpe_37")

        self.verticalLayout_17.addWidget(self.golpe_37)

        self.golpe_38 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_38.setObjectName(u"golpe_38")

        self.verticalLayout_17.addWidget(self.golpe_38)

        self.golpe_39 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_39.setObjectName(u"golpe_39")

        self.verticalLayout_17.addWidget(self.golpe_39)

        self.golpe_40 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_40.setObjectName(u"golpe_40")

        self.verticalLayout_17.addWidget(self.golpe_40)

        self.golpe_41 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_41.setObjectName(u"golpe_41")

        self.verticalLayout_17.addWidget(self.golpe_41)

        self.golpe_42 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_42.setObjectName(u"golpe_42")

        self.verticalLayout_17.addWidget(self.golpe_42)


        self.horizontalLayout_5.addLayout(self.verticalLayout_17)

        self.verticalLayout_22 = QVBoxLayout()
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.label_60 = QLabel(self.verticalLayoutWidget_5)
        self.label_60.setObjectName(u"label_60")

        self.verticalLayout_22.addWidget(self.label_60)

        self.label_61 = QLabel(self.verticalLayoutWidget_5)
        self.label_61.setObjectName(u"label_61")

        self.verticalLayout_22.addWidget(self.label_61)

        self.label_62 = QLabel(self.verticalLayoutWidget_5)
        self.label_62.setObjectName(u"label_62")

        self.verticalLayout_22.addWidget(self.label_62)

        self.label_63 = QLabel(self.verticalLayoutWidget_5)
        self.label_63.setObjectName(u"label_63")

        self.verticalLayout_22.addWidget(self.label_63)

        self.label_64 = QLabel(self.verticalLayoutWidget_5)
        self.label_64.setObjectName(u"label_64")

        self.verticalLayout_22.addWidget(self.label_64)

        self.label_65 = QLabel(self.verticalLayoutWidget_5)
        self.label_65.setObjectName(u"label_65")

        self.verticalLayout_22.addWidget(self.label_65)

        self.label_66 = QLabel(self.verticalLayoutWidget_5)
        self.label_66.setObjectName(u"label_66")

        self.verticalLayout_22.addWidget(self.label_66)

        self.label_67 = QLabel(self.verticalLayoutWidget_5)
        self.label_67.setObjectName(u"label_67")

        self.verticalLayout_22.addWidget(self.label_67)

        self.label_68 = QLabel(self.verticalLayoutWidget_5)
        self.label_68.setObjectName(u"label_68")

        self.verticalLayout_22.addWidget(self.label_68)

        self.label_69 = QLabel(self.verticalLayoutWidget_5)
        self.label_69.setObjectName(u"label_69")

        self.verticalLayout_22.addWidget(self.label_69)

        self.label_70 = QLabel(self.verticalLayoutWidget_5)
        self.label_70.setObjectName(u"label_70")

        self.verticalLayout_22.addWidget(self.label_70)

        self.label_71 = QLabel(self.verticalLayoutWidget_5)
        self.label_71.setObjectName(u"label_71")

        self.verticalLayout_22.addWidget(self.label_71)

        self.label_72 = QLabel(self.verticalLayoutWidget_5)
        self.label_72.setObjectName(u"label_72")

        self.verticalLayout_22.addWidget(self.label_72)

        self.label_75 = QLabel(self.verticalLayoutWidget_5)
        self.label_75.setObjectName(u"label_75")

        self.verticalLayout_22.addWidget(self.label_75)


        self.horizontalLayout_5.addLayout(self.verticalLayout_22)

        self.verticalLayout_18 = QVBoxLayout()
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.golpe_43 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_43.setObjectName(u"golpe_43")

        self.verticalLayout_18.addWidget(self.golpe_43)

        self.golpe_44 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_44.setObjectName(u"golpe_44")

        self.verticalLayout_18.addWidget(self.golpe_44)

        self.golpe_45 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_45.setObjectName(u"golpe_45")

        self.verticalLayout_18.addWidget(self.golpe_45)

        self.golpe_46 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_46.setObjectName(u"golpe_46")

        self.verticalLayout_18.addWidget(self.golpe_46)

        self.golpe_47 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_47.setObjectName(u"golpe_47")

        self.verticalLayout_18.addWidget(self.golpe_47)

        self.golpe_48 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_48.setObjectName(u"golpe_48")

        self.verticalLayout_18.addWidget(self.golpe_48)

        self.golpe_49 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_49.setObjectName(u"golpe_49")

        self.verticalLayout_18.addWidget(self.golpe_49)

        self.golpe_50 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_50.setObjectName(u"golpe_50")

        self.verticalLayout_18.addWidget(self.golpe_50)

        self.golpe_51 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_51.setObjectName(u"golpe_51")

        self.verticalLayout_18.addWidget(self.golpe_51)

        self.golpe_52 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_52.setObjectName(u"golpe_52")

        self.verticalLayout_18.addWidget(self.golpe_52)

        self.golpe_53 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_53.setObjectName(u"golpe_53")

        self.verticalLayout_18.addWidget(self.golpe_53)

        self.golpe_54 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_54.setObjectName(u"golpe_54")

        self.verticalLayout_18.addWidget(self.golpe_54)

        self.golpe_55 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_55.setObjectName(u"golpe_55")

        self.verticalLayout_18.addWidget(self.golpe_55)

        self.golpe_56 = QLineEdit(self.verticalLayoutWidget_5)
        self.golpe_56.setObjectName(u"golpe_56")

        self.verticalLayout_18.addWidget(self.golpe_56)


        self.horizontalLayout_5.addLayout(self.verticalLayout_18)


        self.verticalLayout_10.addLayout(self.horizontalLayout_5)

        self.finalizar_button = QPushButton(self.verticalLayoutWidget_5)
        self.finalizar_button.setObjectName(u"finalizar_button")

        self.verticalLayout_10.addWidget(self.finalizar_button)


        self.horizontalLayout_4.addWidget(self.frame_3)

        StackedWidget.addWidget(self.golpes)

        self.retranslateUi(StackedWidget)

        StackedWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(StackedWidget)
    # setupUi

    def retranslateUi(self, StackedWidget):
        StackedWidget.setWindowTitle(QCoreApplication.translate("StackedWidget", u"StackedWidget", None))
        self.label.setText(QCoreApplication.translate("StackedWidget", u"Adicione o nome da planilha a ser criada", None))
        self.adicionar_planilha_button.setText(QCoreApplication.translate("StackedWidget", u"Adicionar Planilha", None))
        self.label_2.setText(QCoreApplication.translate("StackedWidget", u"Formul\u00e1rio da Planilha Bate Estaca", None))
        self.label_3.setText(QCoreApplication.translate("StackedWidget", u"nome da aba", None))
        self.label_5.setText(QCoreApplication.translate("StackedWidget", u"cliente:", None))
        self.label_8.setText(QCoreApplication.translate("StackedWidget", u"obra:", None))
        self.label_10.setText(QCoreApplication.translate("StackedWidget", u"local:", None))
        self.label_17.setText(QCoreApplication.translate("StackedWidget", u"area:", None))
        self.label_11.setText(QCoreApplication.translate("StackedWidget", u"nega:", None))
        self.label_9.setText(QCoreApplication.translate("StackedWidget", u"be:", None))
        self.cliente_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.obra_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.local_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.nega_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.be_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.estaca_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.estaca_button_2.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.label_7.setText(QCoreApplication.translate("StackedWidget", u"estaca:", None))
        self.label_6.setText(QCoreApplication.translate("StackedWidget", u"secao:", None))
        self.label_12.setText(QCoreApplication.translate("StackedWidget", u"data inicial:", None))
        self.label_13.setText(QCoreApplication.translate("StackedWidget", u"data final:", None))
        self.label_14.setText(QCoreApplication.translate("StackedWidget", u"comp. cravado:", None))
        self.label_15.setText(QCoreApplication.translate("StackedWidget", u"peso martelo:", None))
        self.label_16.setText(QCoreApplication.translate("StackedWidget", u"altura da queda:", None))
        self.secao_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.data_inicial_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.data_final_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.comp_cravado_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.peso_martelo_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.altura_queda_button.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.altura_queda_button_2.setText(QCoreApplication.translate("StackedWidget", u"adicionar", None))
        self.salvar_button.setText(QCoreApplication.translate("StackedWidget", u"salvar", None))
        self.cancelar_button.setText(QCoreApplication.translate("StackedWidget", u"cancelar", None))
        self.label_4.setText(QCoreApplication.translate("StackedWidget", u"Golpes", None))
        self.label_18.setText(QCoreApplication.translate("StackedWidget", u"1", None))
        self.label_27.setText(QCoreApplication.translate("StackedWidget", u"2", None))
        self.label_28.setText(QCoreApplication.translate("StackedWidget", u"3", None))
        self.label_30.setText(QCoreApplication.translate("StackedWidget", u"4", None))
        self.label_29.setText(QCoreApplication.translate("StackedWidget", u"5", None))
        self.label_26.setText(QCoreApplication.translate("StackedWidget", u"6", None))
        self.label_23.setText(QCoreApplication.translate("StackedWidget", u"7", None))
        self.label_24.setText(QCoreApplication.translate("StackedWidget", u"8", None))
        self.label_25.setText(QCoreApplication.translate("StackedWidget", u"9", None))
        self.label_22.setText(QCoreApplication.translate("StackedWidget", u"10", None))
        self.label_21.setText(QCoreApplication.translate("StackedWidget", u"11", None))
        self.label_20.setText(QCoreApplication.translate("StackedWidget", u"12", None))
        self.label_74.setText(QCoreApplication.translate("StackedWidget", u"13", None))
        self.label_19.setText(QCoreApplication.translate("StackedWidget", u"14", None))
        self.label_32.setText(QCoreApplication.translate("StackedWidget", u"15", None))
        self.label_33.setText(QCoreApplication.translate("StackedWidget", u"16", None))
        self.label_34.setText(QCoreApplication.translate("StackedWidget", u"17", None))
        self.label_35.setText(QCoreApplication.translate("StackedWidget", u"18", None))
        self.label_36.setText(QCoreApplication.translate("StackedWidget", u"19", None))
        self.label_37.setText(QCoreApplication.translate("StackedWidget", u"20", None))
        self.label_38.setText(QCoreApplication.translate("StackedWidget", u"21", None))
        self.label_39.setText(QCoreApplication.translate("StackedWidget", u"22", None))
        self.label_40.setText(QCoreApplication.translate("StackedWidget", u"23", None))
        self.label_41.setText(QCoreApplication.translate("StackedWidget", u"24", None))
        self.label_42.setText(QCoreApplication.translate("StackedWidget", u"25", None))
        self.label_43.setText(QCoreApplication.translate("StackedWidget", u"26", None))
        self.label_44.setText(QCoreApplication.translate("StackedWidget", u"27", None))
        self.label_45.setText(QCoreApplication.translate("StackedWidget", u"28", None))
        self.label_46.setText(QCoreApplication.translate("StackedWidget", u"29", None))
        self.label_47.setText(QCoreApplication.translate("StackedWidget", u"30", None))
        self.label_48.setText(QCoreApplication.translate("StackedWidget", u"31", None))
        self.label_49.setText(QCoreApplication.translate("StackedWidget", u"32", None))
        self.label_50.setText(QCoreApplication.translate("StackedWidget", u"33", None))
        self.label_51.setText(QCoreApplication.translate("StackedWidget", u"34", None))
        self.label_52.setText(QCoreApplication.translate("StackedWidget", u"35", None))
        self.label_53.setText(QCoreApplication.translate("StackedWidget", u"36", None))
        self.label_54.setText(QCoreApplication.translate("StackedWidget", u"37", None))
        self.label_55.setText(QCoreApplication.translate("StackedWidget", u"38", None))
        self.label_56.setText(QCoreApplication.translate("StackedWidget", u"39", None))
        self.label_57.setText(QCoreApplication.translate("StackedWidget", u"40", None))
        self.label_58.setText(QCoreApplication.translate("StackedWidget", u"41", None))
        self.label_59.setText(QCoreApplication.translate("StackedWidget", u"42", None))
        self.label_60.setText(QCoreApplication.translate("StackedWidget", u"43", None))
        self.label_61.setText(QCoreApplication.translate("StackedWidget", u"44", None))
        self.label_62.setText(QCoreApplication.translate("StackedWidget", u"45", None))
        self.label_63.setText(QCoreApplication.translate("StackedWidget", u"46", None))
        self.label_64.setText(QCoreApplication.translate("StackedWidget", u"47", None))
        self.label_65.setText(QCoreApplication.translate("StackedWidget", u"48", None))
        self.label_66.setText(QCoreApplication.translate("StackedWidget", u"49", None))
        self.label_67.setText(QCoreApplication.translate("StackedWidget", u"50", None))
        self.label_68.setText(QCoreApplication.translate("StackedWidget", u"51", None))
        self.label_69.setText(QCoreApplication.translate("StackedWidget", u"52", None))
        self.label_70.setText(QCoreApplication.translate("StackedWidget", u"53", None))
        self.label_71.setText(QCoreApplication.translate("StackedWidget", u"54", None))
        self.label_72.setText(QCoreApplication.translate("StackedWidget", u"55", None))
        self.label_75.setText(QCoreApplication.translate("StackedWidget", u"56", None))
        self.finalizar_button.setText(QCoreApplication.translate("StackedWidget", u"Finalizar", None))
    # retranslateUi

