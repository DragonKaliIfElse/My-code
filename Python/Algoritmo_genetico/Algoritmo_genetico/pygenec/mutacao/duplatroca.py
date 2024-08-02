import numpy as np
from numpy.random import randint
from numpy import array

from .mutacao import Mutacao

class DuplaTroca(Mutacao):
	'''
	Mutação dupla troca.
	
	Entrada:
		populacao - Vetor de população que deverá sofrer mutação.
		pmut - Proabilidade de ocorrer uma mutação.
	'''
	def __init__(self, pmut):
		super(DuplaTroca, self).__init__(pmut)
		
	def mutacao(self):
		'''Alteração genética de membros da população usando dupla troca.'''
		nmut = self.selecao()
		
		gen1 = array([randint(0, self.ngen-1)])
		gen2 = array([randint(0, self.ngen-1)])
		
		self.populacao[nmut, gen1], self.populacao[nmut, gen2] = self.populacao[nmut, gen2], self.populacao[nmut, gen1]
