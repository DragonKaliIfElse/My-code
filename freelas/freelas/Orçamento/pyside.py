import sys
import os # Importe o módulo os
from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader

# ... (função load_ui_file)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Define o caminho do arquivo UI de forma relativa ao script
    ui_file_path = os.path.join(
        os.path.dirname(__file__), # Pega o diretório do script atual
        "minha_janela.ui"          # Adiciona o nome do arquivo
    )
    
    sys.exit(app.exec())
