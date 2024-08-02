import numpy as np
from numpy.random import randint
from numpy import array

from .mutacao import Mutacao

class Flip(Mutacao):
	''' 
	Mutação flip.
	
	Entrada:
		pmut - Probabilidade de ocorrer uma mutação.
	'''
	def __init__(self, pmut):
		super(Flip, self).__init__(pmut)
		
	def mutacao(self):
		'''Alteração genética de membros da populaçãou usando mutação flip.'''
		nmut = self.selecao()
		genflip = array([randint(0, self.ngen-1) for _ in nmut])
		self.populacao[nmut, genflip] = 1-self.populacao[nmut, genflip]
		
