from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Configurações do Firefox
options = webdriver.FirefoxOptions()
options.add_argument("--headless")  # roda sem abrir janela
driver = webdriver.Firefox(options=options)

def find_description(URL):
    variaveis = {}
    variaveis2 = {}
    driver.get(URL)
    time.sleep(2)  # espera carregar (ajuste se o site for lento)
    '''
    '''
    xpath0 = f"/html/body/div/section/div[2]/div[2]/div"
    elementos0 = driver.find_elements(By.XPATH, xpath0)
    v_max = len(elementos0)
    for v in range(1, v_max+1):
        xpath = f"/html/body/div/section/div[2]/div[2]/div[{v}]/section/section/div/div"
        try:
            elementos_div1 = driver.find_elements(By.XPATH, xpath)
            x_max = len(elementos_div1)
        except Exception as e:
            print(f"erro ao achar primeira div: {e}")
            return 1
        for x in range(1, x_max + 1):
            try:
                xpath_button = f"{xpath}[{x}]/span/span[1]/span/span"
                botao = driver.find_element(By.XPATH, xpath_button)
                botao.click()
            except:
                xpath_button = f"{xpath}[{x}]/span/div/span/span/span[1]/span/span"
                botao = driver.find_element(By.XPATH, xpath_button)
                botao.click()

            finally:
                xpath_lines = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr"
                elementos = driver.find_elements(By.XPATH, xpath_lines)
                y_max = len(elementos)
                for y in range(1, y_max+1):
                    try:
                        xpath_var = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[1]"
                        var = driver.find_element(By.XPATH, xpath_var)
                        try:
                            xpath_description = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/div/p"
                            description = driver.find_element(By.XPATH, xpath_description)
                            variaveis[var.text] = description.text
                        except:
                            xpath_description_button = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/span[1]/span"
                            botao_description = driver.find_element(By.XPATH, xpath_description_button)
                            botao_description.click()
                            try:
                                xpath_description_button2 = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/span[2]/span/span/span[1]/span/span[2]"
                                botao_description2 = driver.find_element(By.XPATH, xpath_description_button2)
                                botao_description2.click()
                                xpath_lines2 = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/span[2]/span/span/span[5]/table/tbody/tr"
                                elementos2 = driver.find_elements(By.XPATH, xpath_lines2)
                                z_max = len(elementos2)
                                for z in range(1, z_max+1):
                                    xpath_var2 = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/span[2]/span/span/span[5]/table/tbody/tr[{z}]/td[1]"
                                    var2 = driver.find_element(By.XPATH, xpath_var2)
                                    xpath_description2 = f"{xpath}[{x}]/span/div/span/span/span[4]/table/tbody/tr[{y}]/td[2]/span/span/span[2]/span/span/span[5]/table/tbody/tr[{z}]/td[2]/span/span/div/p"
                                    try:
                                        description2 = driver.find_element(By.XPATH, xpath_description2)
                                        variaveis2[var2.text] = description2.text
                                    except:
                                        continue
                                variaveis[var.text] = variaveis2

                            except:
                                continue


                    except:
                        continue

    # Filtra valores vazios (se necessário)
    print("Valores coletados:", variaveis)

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
