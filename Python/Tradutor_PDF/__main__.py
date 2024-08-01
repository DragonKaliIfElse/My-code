from googletrans import Translator
import pyautogui
import pytesseract
import time
import fitz
import numpy as np
import os
import io
from PIL import Image

class Tradutor:
	def __init__(self, nome_pdf, pagina_inicial, pagina_final, alfabeto, tipo):
		self.nome_pdf = nome_pdf
		self.pagina_inicial = pagina_inicial
		self.pagina_final = pagina_final
		self.tipo = tipo
		self.alfabeto = 'eng'

	def set_nome_pdf(self):
		self.nome_pdf = input('Qual o nome do pdf a ser traduzido?(deve estar contido na pasta LIVROS)\n')
		self.nome_pdf = f'/home/dragon/Python/Tradutor_PDF/LIVROS/{self.nome_pdf}'
		self.meu_arquivo = self.nome_pdf.replace('.pdf','.txt')

	def set_pagina_inicial(self):
		 pag = input('Qual a primeira página a ser traduzida?\n')
		 self.pagina_inicial = int(pag)

	def set_pagina_final(self):
		 pag = input('Qual a última página a ser traduzida?\n')
		 self.pagina_final = int(pag)

	def set_tipo(self):
		self.tipo = input('Qual o tipo do pdf, digitalizado(dgt) ou textual(txt)?\n')
	def set_alfabeto(self):
		self.alfabeto = input('Qual o alfabeto do arquivo? russo(rus)\n')
		if self.alfabeto == '': self.alfabeto = 'eng';

	def le_pagina_image(self,x): # Le o texto da página usando um escaneador de imagem
		pdf_document = fitz.open(self.nome_pdf)
		page = pdf_document.load_page(x)  # Especifica a pagina do PDF
		page_image = page.get_images(full=True)  # Extraia o texto da página
		texts=[]
		for img_index, img in enumerate(page_image):
			xref = img[0]
			base_image = pdf_document.extract_image(xref)
			image_bytes = base_image['image'] 
			image_ext = base_image['ext']
			image = Image.open(io.BytesIO(image_bytes))
			text = pytesseract.image_to_string(image, lang=self.afabeto)
			texts.append(text)
		pdf_document.close()
		return text

	def le_pagina_pdf(self,x): # Le o texto da página se for do tipo .pdf
		pdf_document = fitz.open(self.nome_pdf)
		page = pdf_document.laod_page(x)  # Especifica a pagina do PDF
		text = page.get_text()  # Extraia o texto da página
		return text

	def translate_text(self, text, target_language='pt'): # Traduz texto para a lingua selecioanada
		translator = Translator()
		translation = translator.translate(text, dest=target_language)
		return translation.text

	def redige_texto(self, text): # Vetoriza o texto, separa o arquivo por linhas, concatena elas e replace('.','.\n')
		contador = 0
		for char in text:
			if contador == 0 or contador == 1:
				print(f'> {char}')
				contador+=1

		return text

	def traduz_pdf(self): #
		pagina = self.pagina_inicial-1
		ultima_pagina = self.pagina_final-1
		contador = 0

		media_pag = (pagina + ultima_pagina)/
		if self.tipo == 'txt': teste = self.le_pagina_pdf(int(media_pag)); teste = self.redige_texto(teste);
		elif self.tipo == 'dgt': teste = self.le_pagina_image(int(media_pag)); teste = self.redige_texto(teste);
		print(f'\nTeste:\n{teste}\n\npágina: {int(media_pag+1)}\n')
		resposta = input('O teste foi satisfatório?\n')
		resposta = resposta.lower()

		if resposta == 'sim' or 's' or 'yes' or 'y':
			print(f'Executando código...\n')

		elif resposta == 'não' or 'n' or 'no':
			print('Finalizando código.\n')
			return None

		else:
			print('Reinicie o programa, e escolha sim ou não nesta etapa')
			return None

		with open(self.meu_arquivo, 'w') as file:

			while pagina <= ultima_pagina:
				if self.tipo == 'txt':
					text = self.le_pagina_pdf(pagina)
				elif self.tipo == 'dgt':
					text = self.le_pagina_image(pagina)

				if text == '':
					pagina+=1
					continue
				text = self.redige_texto(text)
				text_traduzido = self.translate_text(text)
				progresso = (pagina+1-self.pagina_inicial)/(ultima_pagina+1-self.pagina_inicial)*100
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


				file.write(f'\n{text_traduzido}\n\npagina: {pagina+1}\n')
				pagina+=1
				contador+=1

		return 0
def main():
	texto = Tradutor('', '', '', '', '')
	texto.set_nome_pdf()
	texto.set_pagina_inicial()
	texto.set_pagina_final()
	texto.set_tipo()
	texto.set_alfabeto()

	programa = texto.traduz_pdf()
	if programa == 0:
		print(f'\n***FIM DA TRADUÇÃO***\n')

if __name__ == "__main__": main();
'''
🌑🌒🌓🌔🌕🌖🌗🌘
'''
