from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Configurações do Firefox
options = webdriver.FirefoxOptions()
options.add_argument("--headless")  # roda sem abrir janela
driver = webdriver.Firefox(options=options)

def find_description(URL):
    driver.get(URL)
    time.sleep(2)  # espera carregar (ajuste se o site for lento)
 
    file = open('vast_ai.txt', 'w')
    file.close()
    file = open('vast_ai.txt', 'a')

    xpath_areas = f"/html/body/div[2]/div/div[1]/div[1]/div[2]/div/div[2]/div/a"
    areas = driver.find_elements(By.XPATH, xpath_areas)
    x_max = len(areas)
    for x in range(1, x_max+1):
        xpath_area = f"{xpath_areas}[{x}]"
        area = driver.find_element(By.XPATH, xpath_area)
        file.write(f'{area.text}:\n')
        area.click()
        xpath_sessoes = f"/html/body/div[2]/div/div[1]/div[2]/div[1]/div/div/div[2]/div"
        sessoes = driver.find_elements(By.XPATH, xpath_sessoes)
        y_max = len(sessoes)
        for y in range(1, y_max+1):
            xpath_sessao = f"{xpath_sessoes}[{y}]"
            xpath_subsessoes = f"{xpath_sessao}/ul/li"
            subsessoes = driver.find_elements(By.XPATH, xpath_subsessoes)
            z_max = len(subsessoes)
            for z in range(1, z_max+1):
                subsubsessao=''
                try:
                    xpath_subsessao = f"{xpath_subsessoes}[{z}]/a"
                    subsessao = driver.find_element(By.XPATH, xpath_subsessao)
                    file.write(f'\t{subsessao.text}:\n')
                    print(subsessao.text)
                    subsessao.click()
                    time.sleep(1)
                    try:
                        xpath_content = '//*[@id="api-playground-2-operation-page"]'
                        content = driver.find_element(By.XPATH, xpath_content)
                        file.write(f'<content>{content.text}</content>\n')
                    except:
                        xpath_content = '//*[@id="content"]'
                        content = driver.find_element(By.XPATH, xpath_content)
                        file.write(f'<content>{content.text}</content>\n')
                except:
                    xpath_subsessao = f"{xpath_subsessoes}[{z}]/button"
                    subsessao = driver.find_element(By.XPATH, xpath_subsessao)
                    file.write(f'\t{subsessao.text}:\n')
                    subsessao.click()
                    xpath_subsubsessoes = f"{xpath_sessao}/ul/li[{z}]/ul/li"
                    subsubsessoes = driver.find_elements(By.XPATH, xpath_subsubsessoes)
                    a_max = len(subsubsessoes)
                    for a in range(1, a_max+1):
                        try:
                            xpath_subsubsessao = f"{xpath_subsubsessoes}[{a}]/a"
                            subsubsessao = driver.find_element(By.XPATH, xpath_subsubsessao)
                            file.write(f'\t\t{subsubsessao.text}:\n')
                            print(subsubsessao.text)
                            subsubsessao.click()
                            time.sleep(1)
                            try:
                                xpath_content = '//*[@id="api-playground-2-operation-page"]'
                                content = driver.find_element(By.XPATH, xpath_content)
                                file.write(f'<content>{content.text}</content>\n')
                            except:
                                xpath_content = '//*[@id="content"]'
                                content = driver.find_element(By.XPATH, xpath_content)
                                file.write(f'<content>{content.text}</content>\n')
                        except:
                            xpath_subsubsessao = f"{xpath_subsubsessoes}[{a}]/button"
                            subsubsessao = driver.find_element(By.XPATH, xpath_subsubsessao)
                            file.write(f'\t\t{subsubsessao.text}:\n')
                            print(subsubsessao.text)
                            subsubsessao.click()
                            xpath_subsubsubsessoes = f"{xpath_subsubsessoes}[{a}]/ul/li"
                            subsubsubsessoes = driver.find_elements(By.XPATH, xpath_subsubsubsessoes)
                            b_max = len(subsubsubsessoes)
                            for b in range(1, b_max+1):
                                xpath_subsubsubsessao = f'{xpath_subsubsubsessoes}[{b}]/a'
                                subsubsubsessao = driver.find_element(By.XPATH, xpath_subsubsubsessao)
                                print(subsubsubsessao.text)
                                subsubsubsessao.click()
                                time.sleep(1)
                                try:
                                    xpath_content = '//*[@id="api-playground-2-operation-page"]'
                                    content = driver.find_element(By.XPATH, xpath_content)
                                    file.write(f'<content>{content.text}</content>\n')
                                except:
                                    xpath_content = '//*[@id="content"]'
                                    content = driver.find_element(By.XPATH, xpath_content)
                                    file.write(f'<content>{content.text}</content>\n')
    file.close()
                    

def main():
    try:
        # 👉 Coloca aqui a URL do site
        URL = "https://docs.vast.ai/documentation/get-started"
        find_description(URL)

    finally:
        driver.quit()
if __name__ == '__main__': main();
