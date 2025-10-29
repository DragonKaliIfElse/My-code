from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Configurações do Firefox
options = webdriver.FirefoxOptions()
options.add_argument("--headless")  # roda sem abrir janela
driver = webdriver.Firefox(options=options)

def find_description(URL):
    valores_coletados = []
    driver.get(URL)
    time.sleep(2)  # espera carregar (ajuste se o site for lento)
 
    xpath1 = f"/html/body/div/section/div[2]/div[2]/div"
    try:
        elementos_div1 = driver.find_elements(By.XPATH, xpath1)
    except Exception as e:
        print(f"erro ao achar primeira div: {e}")
        return 1
    x_max = len(elementos_div1)
    for x in range(1, x_max + 1):
        xpath2 = f"/html/body/div/section/div[2]/div[2]/div[{x}]/section/div/span"
        try:
            elementos_div2 = driver.find_elements(By.XPATH, xpath2)
        except:
            continue
        y_max = len(elementos_div2)
        for y in range(1, y_max + 1):
            xpath3 = f"/html/body/div/section/div[2]/div[2]/div[{x}]/section/div/span[{y}]/div/div/span"
            try:
                elementos_div3 = driver.find_elements(By.XPATH, xpath3)
            except:
                continue
            z_max = len(elementos_div3)
            for z in range(1, z_max + 1):
                try:
                    # Constroi o XPath com os valores atuais da iteração
                    xpath = f"/html/body/div/section/div[2]/div[2]/div[{x}]/section/div/span[{y}]/div/div/span[{z}]/div/div/div"
                    # Tenta encontrar o elemento
                    elemento = driver.find_element("xpath", xpath)
                    valores_coletados.append(elemento.text)
                except:
                    continue

    # Filtra valores vazios (se necessário)
    valores_coletados = [v for v in valores_coletados if v.strip()]

    print("Valores coletados:", valores_coletados)

def main():
    try:
        # 👉 Coloca aqui a URL do site
        URL = "https://api.sienge.com.br/docs/"
        driver.get(URL)
        time.sleep(2)  # espera carregar (ajuste se o site for lento)

        # Pega todos os <a> dentro de <li>
        links = driver.find_elements(By.CSS_SELECTOR, "li a")
        hrefs = [link.get_attribute("href") for link in links if link.get_attribute("href")]

        # Exibe resultados
        for href in hrefs:
            print(href)
            find_description(href)

        print(f"\nTotal de links encontrados: {len(hrefs)}")

    finally:
        driver.quit()
if __name__ == '__main__': main();
