import numpy as np
from numpy.random import random
from numpy import array

from .selecao import Selecao

class Roleta(Selecao):
	"""
	Seleciona indivíduos para cruzamento usando roleta de seleção.
	Recebe como entrada:
		populacao - Objeto criado a partir da classe Populacao.
	"""
	def __init__(self, populacao, m='max'):
		super(Roleta, self).__init__(populacao)
		self.m=m
		
	def selecionar(self, fitness):
		"""Roleta de seleção de indivíduos"""
		if fitness is None:
			fitness = self.populacao.avaliar()
		fmin = fitness.min()
		fitness = fitness - fmin
		if self.contador == 0:
			print(f'fitness\n{fitness}\n')
		total = fitness.sum()
		parada = total * (1.0 - random())
		print(f'parada\n{parada}\n')
		parcial = 0
		i = 0
		j = 0
		contador = 0
		
		for linha in range(fitness.shape[0]):
			if parcial < parada:
				for coluna in range(fitness.shape[1]):	
					parcial += fitness[linha,coluna]
					contador += 1
					if parcial >= parada:
						i = linha
						j = coluna
						print(f'parcial\n{parcial}\nfitness[i,j]\n{fitness[i,j]}\ncontador\n{contador}\n')
						break
			else:
					break
		
		self.contador += 1
		return i
		
	 
