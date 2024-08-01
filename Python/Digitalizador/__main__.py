]import pyautogui 
import pytesseract
import time
import PyPDF2
import numpy as np
import os

class Digitalizador:
	def __init__(self, nome_pdf, pagina_final, meu_arquivo):
		self.nome_pdf = nome_pdf
		self.pagina_final = pagina_final
		self.meu_arquivo = meu_arquivo
		
	def set_nome_pdf(self):
		self.nome_pdf = input('Qual o diretório ou nome do pdf a ser traduzido?\n')
	
	def set_pagina_inicial(self):
		 pag = input('Qual a primeira página a ser traduzida?\n')
		 self.pagina_inicial = int(pag)
		
	def set_pagina_final(self):
		 pag = input('Qual a última página a ser traduzida?\n')
		 self.pagina_final = int(pag)
		 
	def set_meu_arquivo(self):
		self.meu_arquivo = input('Qual o nome do novo texto traduzido?\n')
	
	def le_pagina(self): # Le o texto da página usando um escaneador de imagem

		region = (178, 55, 1005, 707)
		image = pyautogui.screenshot(region = region)
		text = pytesseract.image_to_string(image).strip()
		return text	
		
	def abre_arquivo(self):
		pyautogui.hotkey('ctrl', 'shift', 'd')
		time.sleep(0.5)
		pyautogui.write("open ")
		pyautogui.write(self.nome_pdf)
		pyautogui.hotkey('tab')
		pyautogui.hotkey('enter')		
	
	def digitaliza_pdf(self): #
		self.abre_arquivo()
		
		pagina = 1
		contador = 0
			
		with open(self.meu_arquivo, 'w') as file:
		
			while pagina <= self.pagina_final:
				texto = self.le_pagina()
				if texto == '':
					pagina+=1
					continue
				progresso = (pagina/self.pagina_final)*100
				if contador == 0:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌑 pagina: {pagina+1}\n')
				if contador == 1:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌒 pagina: {pagina+1}\n')
				if contador == 2:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌓 pagina: {pagina+1}\n')
				if contador == 3:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌔 pagina: {pagina+1}\n')
				if contador == 4:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌕 pagina: {pagina+1}\n')
				if contador == 5:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌖 pagina: {pagina+1}\n')
				if contador == 6:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌗 pagina: {pagina+1}\n')
				if contador == 7:
					os.system('clear')
					print(f'\nprogresso: {progresso:.2f}% 🌘 pagina: {pagina+1}\n')
					contador = -1
				
					
				file.write(f'\n{texto}\n\npagina: {pagina+1}\n')
				pyautogui.hotkey('pagedown')
				pyautogui.hotkey('pagedown')
				pagina+=1
				contador+=1
				
		return 0
		
def main()
	texto = Digitalizador(None, None, None)
	texto.set_nome_pdf()
	texto.set_pagina_final()
	texto.set_meu_arquivo()

	programa = texto.digitaliza_pdf()
	if programa == 0:
		print(f'\n***FIM DA DIGITALIZAÇÃO***\n')	

	'''
	🌑🌒🌓🌔🌕🌖🌗🌘
	'''
if __name__ == "__main__":
	main()





