from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import time

#Inicializando WebDriver
service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
url = "https://www.playpix.com/pb/?openGames=40003094-real&gameNames=Roulette%20A"
driver.get(url)

class WebBot:
	def __init__(self):
		self.enableBet = False
		self.colected = False
		self.triggered = 0
	def ValorDaRoleta(self):
		element = driver.find_element(By.CSS_SELECTOR, "#root > div.layout-content-holder-bc > div.casino-full-game-bg > div.casino-full-game-container.num-casino-games-1 > div > iframe")
		driver.switch_to.frame(element)#mudando o contexto para iframe
		element = driver.find_element(By.CSS_SELECTOR, "#root > div > div.iframe-holder > div > iframe")
		driver.switch_to.frame(element)#mudando o contexto para iframe2
		element = driver.find_element(By.CSS_SELECTOR, "#app > div > div.roulette-desktop.theme-4.live-casino-game-wrapper")
		elementTeste1 = element.find_elements(By.CLASS_NAME, "board-wrapper-layout.board-wrapper-lied")
		elementTeste2 = element.find_elements(By.CLASS_NAME, "board-wrapper-layout")
		if elementTeste1:
			self.enableBet = False
			self.triggered = 0
			self.colected = False
			driver.switch_to.default_content()
			return None
		elif elementTeste2 and self.triggered == 0:
			self.enableBet = True
			self.triggered +=1
			self.colected = True
			time.sleep(3)
			try:
				element = driver.find_element(By.CSS_SELECTOR, "#app > div > div.roulette-desktop.theme-4.live-casino-game-wrapper > div.board-wrapper-layout > div.board-wrapper-right > div > div.mini-statistics > div:nth-child(1) > span")
				valorDaRoleta = element.text
				driver.switch_to.default_content()
				return valorDaRoleta
			except self.NoSuchElementException as erro:
				print(f"Error: {erro}\n")
				self.triggered -=1
				self.enableBet = False
				self.colected = False
				driver.switch_to.default_content()
				pass
		elif elementTeste2 and self.colected == True:
			self.triggered +=1
			self.colected = False
			driver.switch_to.default_content()
			return None
		else:
			driver.switch_to.default_content()
			return None

	def getEnableBet(self):
		return self.enableBet
	def getTriggered(self):
		return self.triggered
	def getColected(self):
		return self.colected

	class NoSuchElementFounded(Exception):
		def __init__(self, mensagem="Nenhum elemento encontrado."):
			super().__init__(mensagem)
