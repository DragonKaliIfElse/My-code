import numpy as np
from numpy.random import choice
from numpy import array, where

from .selecao import Selecao

class Torneio(Selecao):
	'''
	Seleciona indivíduos para cruzamento usando Torneio.
	Recebe como entrada:
		populacao - Objeto criado a partir da classe Populacao.
	'''
	def __init__(self, populacao, tamanho=10, m='max'):
		super(Torneio, self).__init__(populacao)
		self.tamanho = tamanho
		self.m=m

	def selecionar(self, fitness=None):
		'''Retorna o indivíduo campeão da rodada.'''
		if fitness is None:
			fitness = self.populacao.avaliar()
		if self.contador == 0:
			print(f'fitness\n{fitness}\n')
		ind = choice(fitness.shape[0], size=self.tamanho, replace=False)
#		print(f'ind\n{ind}\n')
		grupo = fitness[ind]
#		print(f'grupo\n{grupo}\n')
		campeao = grupo.max()
		print(f'campeão\n{campeao}\n')
		i = where(fitness == campeao)[0][0]
#		print(f'i\n{i}\n')
		self.contador +=1
		return i
