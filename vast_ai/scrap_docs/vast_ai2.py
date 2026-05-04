from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

# Configurações do Firefox
options = webdriver.FirefoxOptions()
options.add_argument("--headless")  # roda sem abrir janela
driver = webdriver.Firefox(options=options)

def extract_md_content(url):
    """Extrai conteúdo da URL com .md adicionado usando requisição HTTP"""
    # Normaliza a URL para adicionar .md corretamente
    md_url = url.rstrip('/') + '.md'
    try:
        response = requests.get(md_url, timeout=15)
        return response.text
    except:
        return ""

def find_description(URL):
    driver.get(URL)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div[1]/div[1]/div[2]/div/div[2]/div/a")))
 
    file = open('vast_ai.txt', 'w')
    file.close()
    file = open('vast_ai.txt', 'a')

    xpath_areas = f"/html/body/div[2]/div/div[1]/div[1]/div[2]/div/div[2]/div/a"
    areas = driver.find_elements(By.XPATH, xpath_areas)
    x_max = len(areas)
    for x in range(1, x_max+1):
        xpath_area = f"{xpath_areas}[{x}]"
        area = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_area)))
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
                    subsessao = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_subsessao)))
                    file.write(f'\t{subsessao.text}:\n')
                    print(subsessao.text)
                    subsessao.click()
                    wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="api-playground-2-operation-page"]|//*[@id="content"]')))
                    try:
                        # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                        md_content = extract_md_content(driver.current_url)
                        file.write(f'<content>{md_content}</content>\n')
                    except:
                        # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                        md_content = extract_md_content(driver.current_url)
                        file.write(f'<content>{md_content}</content>\n')
                except:
                    xpath_subsessao = f"{xpath_subsessoes}[{z}]/button"
                    subsessao = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_subsessao)))
                    file.write(f'\t{subsessao.text}:\n')
                    subsessao.click()
                    xpath_subsubsessoes = f"{xpath_sessao}/ul/li[{z}]/ul/li"
                    subsubsessoes = driver.find_elements(By.XPATH, xpath_subsubsessoes)
                    a_max = len(subsubsessoes)
                    for a in range(1, a_max+1):
                        try:
                            xpath_subsubsessao = f"{xpath_subsubsessoes}[{a}]/a"
                            subsubsessao = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_subsubsessao)))
                            file.write(f'\t\t{subsubsessao.text}:\n')
                            print(subsubsessao.text)
                            subsubsessao.click()
                            wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="api-playground-2-operation-page"]|//*[@id="content"]')))
                            try:
                                # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                                md_content = extract_md_content(driver.current_url)
                                file.write(f'<content>{md_content}</content>\n')
                            except:
                                # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                                md_content = extract_md_content(driver.current_url)
                                file.write(f'<content>{md_content}</content>\n')
                        except:
                            xpath_subsubsessao = f"{xpath_subsubsessoes}[{a}]/button"
                            subsubsessao = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_subsubsessao)))
                            file.write(f'\t\t{subsubsessao.text}:\n')
                            subsubsessao.click()
                            xpath_subsubsubsessoes = f"{xpath_subsubsessoes}[{a}]/ul/li"
                            subsubsubsessoes = driver.find_elements(By.XPATH, xpath_subsubsubsessoes)
                            b_max = len(subsubsubsessoes)
                            for b in range(1, b_max+1):
                                xpath_subsubsubsessao = f'{xpath_subsubsubsessoes}[{b}]/a'
                                subsubsubsessao = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, xpath_subsubsubsessao)))
                                print(subsubsubsessao.text)
                                subsubsubsessao.click()
                                wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="api-playground-2-operation-page"]|//*[@id="content"]')))
                                try:
                                    # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                                    md_content = extract_md_content(driver.current_url)
                                    file.write(f'<content>{md_content}</content>\n')
                                except:
                                    # SUBSTITUIÇÃO: extrair conteúdo via URL + .md em vez de XPath
                                    md_content = extract_md_content(driver.current_url)
                                    file.write(f'<content>{md_content}</content>\n')
    file.close()
                    

def main():
    try:
        # 👉 Coloca aqui a URL do site
        URL = "https://docs.vast.ai/documentation/get-started"
        find_description(URL)

    finally:
        driver.quit()
if __name__ == '__main__': main();
