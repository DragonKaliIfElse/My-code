from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Configurações do Firefox
options = webdriver.FirefoxOptions()
options.add_argument("--headless")  # roda sem abrir janela
driver = webdriver.Firefox(options=options)
valores_coletados = []

URL = "https://api.sienge.com.br/docs/html-files/building-projects-progress-logs-v1.html"
driver.get(URL)
time.sleep(2)  # espera carregar (ajuste se o site for lento)
# Defina limites máximos para x, y e z com base na estrutura da página
x_max = 5  # Ajuste conforme necessário
y_max = 5   # Ajuste conforme necessário
z_max = 5   # Ajuste conforme necessário

for x in range(1, x_max + 1):
    for y in range(1, y_max + 1):
        for z in range(1, z_max + 1):
            try:
                # Constroi o XPath com os valores atuais da iteração
                xpath = f"/html/body/div/section/div[2]/div[2]/div[{x}]/section/div/span[{y}]/div/div/span[{z}]/div/div/div"
                # Tenta encontrar o elemento
                elemento = driver.find_element("xpath", xpath)
                valores_coletados.append(elemento.text)
            except Exception as erro:
                print(f'deu erro: {erro}')

# Filtra valores vazios (se necessário)
valores_coletados = [v for v in valores_coletados if v.strip()]

print("Valores coletados:", valores_coletados)
driver.quit()
