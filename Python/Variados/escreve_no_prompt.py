import numpy as np
import pyautogui
import time

def conta_linha():
	numero_de_linhas=0
	with open('texto.txt', 'r') as arquivo:
		for linha in arquivo:
			numero_de_linhas +=1
	return numero_de_linhas
	
numero_linha = 0
total_linhas = conta_linha()

pyautogui.hotkey('alt',	'tab')

with open('texto.txt', 'r') as arquivo:
	while numero_linha < total_linhas:
		linha = arquivo.readline()
		pyautogui.typewrite('/cpf1 ', interval=0.1)
		pyautogui.typewrite(linha, interval=0.1)
		time.sleep(2)
		pyautogui.press('enter')
		numero_linha +=1

