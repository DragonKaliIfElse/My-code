import EstruturaDoJogo as EDJ
import ReinforcementLearning
import WebBot as WB
import numpy as np
import time

def fibonnaci(n):
    resultado = ((1+np.sqrt(5))**n - (1-np.sqrt(5))**n)/((2**n)*np.sqrt(5))
    resultado = int(resultado)
    return resultado

def main():
	input("Press [enter] to continue...")
	webbot = WB.WebBot()
	while True:
		valorDaRoleta = webbot.ValorDaRoleta()
		if webbot.getTriggered() == 1 and webbot.getColected() is True:
			with open("valoresDaRoleta.txt", "a") as file:
				file.write(f'{valorDaRoleta}\n')
			"""
			with open("ValoresDaRoleta.txt", "a") as file:
				file.write(f'{valorDaRoleta}\n')
			"""
		else:
			pass
		'''
		estruturaDoJogo = EDJ.EstruturaDoJogo(valorDaRoleta)
		colun1 = estruturaDoJogo.firstColun_()
		colun2 = estruturaDoJogo.seconColun_()
		colun3 = estruturaDoJogo.thirdColun_()
		dozen1 = estruturaDoJogo.firstDozen_()
		dozen2 = estruturaDoJogo.seconDozen_()
		dozen3 = estruturaDoJogo.thirdColun_()
		black = estruturaDoJogo.blackNumbers_()
		red = estruturaDoJogo.redNumbers_()
		even = estruturaDoJogo.even_()
		odd = estruturaDoJogo.odd_()
		zero = estruturaDoJogo.zero_()
		'''
		print(f'\nnúmero da vez: {valorDaRoleta}\n coluna 1 = {colun1}\n coluna 2 = {colun2}\n coluna 3 = {colun3}\n dúzia 1 = {dozen1}\n dúzia 2 = {dozen2}\n dúzia 3 = {dozen3}\n preto = {black}\n vermelho = {red}\n par = {even}\n impar = {odd}\n zero = {zero}\n')
		time.sleep(2)
	return 0

if __name__ == "__main__": main();

