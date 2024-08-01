from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import random
import time

#Inicializando WebDriver
#service = ChromeService(executable_path=ChromeDriverManager().install())
#driver = webdriver.Chrome(service=service)
#url = "https://www.playpix.com/pb/?openGames=40003094-real&gameNames=Roulette%20A"
#driver.get(url)

class Statistics:
	def __init__(self):
		pass
	def GetData(self):
		element = driver.find_element(By.CSS_SELECTOR, "#root > div.layout-content-holder-bc > div.casino-full-game-bg > div.casino-full-game-container.num-casino-games-1 > div > iframe") #Primeira camada do site
		driver.switch_to.frame(element)#mudando o contexto para iframe
		element = driver.find_element(By.CSS_SELECTOR, "#root > div > div.iframe-holder > div > iframe") #Segunda camada do site
		driver.switch_to.frame(element)#mudando o contexto para iframe2
		element = driver.find_element(By.CSS_SELECTOR, "#app > div > div.roulette-desktop.theme-4.live-casino-game-wrapper > div.control-buttons-layout > div:nth-child(5) > div > div > div.g-statistics-popup > div > div > div.popup-body > div > div > div > div.statistics-in.scroll-vertical.scroll-style > div > div") #Tabela de números registrados
		statisticsData = element.text
		driver.switch_to.default_content()
		print(statisticsData)
		return None

	def PseudoData(self, DataSize):
		array=[]
		for i in range(DataSize):
			randomNumber = random.randint(0,36)
			array.append(randomNumber)
		return array

	class NoSuchElementFounded(Exception):
		def __init__(self, mensagem="Nenhum elemento encontrado."):
			super().__init__(mensagem)
