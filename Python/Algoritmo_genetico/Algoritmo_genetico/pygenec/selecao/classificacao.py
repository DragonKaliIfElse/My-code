import numpy as np
from numpy.random import random
from numpy import array, argsort, where

from .selecao import Selecao

class Classificacao(Selecao):
	"""
	Seleciona individuos para cruzamento usando Classificação.
	Recebe como entrada:
		populacao - Objeto criado a partir da classe Populacao.
	"""
	def __init__(self,populacao, m='max'):
		super(Classificacao, self).__init__(populacao)
		self.m = m
	def selecionar(self, fitness=None):
		"""Roleta de seleção de indivíduos."""
		if fitness is None:
			fitness = self.populacao.avaliar()
			
		valores = fitness.flatten()
		
		ind = argsort(valores)

		classificacao = valores[ind]
		ind_clas = argsort(classificacao)+1
		if self.contador == 0:
			print(f'fitness\n{fitness}\nclassificação\n{classificacao}\n{ind_clas}')
		total = ind_clas.sum()
		parada = total * (1.0-random())
		print(f'parada\n{parada}\ntotal\n{total}')
		parcial = 0
		i = 0
		contador = 0
		nao_acertou = True
		while nao_acertou:
			for p in range(classificacao.size):	
				parcial += p
				contador += 1
				if parcial >= parada:
					selecionado = classificacao[p]
					print(f'parcial\n{parcial}\nselecionado\n{selecionado}\ncontador\n{contador}\n')
					nao_acertou = False
					break
			continue	
		i = where(fitness == selecionado)[0][0]
		print(f'i \n{i}\n')
		self.contador += 1
		return i
