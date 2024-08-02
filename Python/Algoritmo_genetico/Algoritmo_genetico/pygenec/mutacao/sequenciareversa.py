import numpy as np
from numpy.random import randint
from numpy import array

from .mutacao import Mutacao

class SequenciaReversa(Mutacao):
	'''
	Mutação Sequência Reversa.
	
	Entrada:
		pmut - Probabilidade de ocorrer uma mutação.
	'''
	def __init__(self, pmut):
		super(SequenciaReversa, self).__init__(pmut)
		
	def mutacao(self):
		'''
		Alteração genética de membros da população usando sequência reversa.
		'''
		nmut = self.selecao()
		if nmut.size != 0:
			for k in nmut:
				i = randint(0, self.ngen-1)
				j = randint(0, self.ngen-1)			
				while i==j:
					j = randint(0, self.ngen-1)
				if i > j:
					i, j = j, i
				self.populacao[k, i:j] = self.populacao[k, i:j][::-1]
