import numpy as np
from numpy.random import shuffle, randint 
from .cruzamento import Cruzamento, NoCompatibleIndividualSize

class Embaralhamento(Cruzamento):
	'''
	Gerador de população via embaralhamento e cruzamento de um ponto.
	
	Entrada:
		tamanho_populacao - Tamanho final da população resultante
	'''
	def __init__(self, tamanho_populacao):
		super(Embaralhamento, self).__init__(tamanho_populacao)
		
	def cruzamento(self, progenitor1, progenitor2):
		'''
		Cruzamento de dois indivíduos via embaralhamento um ponto.
		
		Entrada:
			ind1 - Primeiro indivíduo
			ind2 - Segundo indivíduo

		O tamanho de ambos os indivíduos deve ser igual, do contrário um erro será levantado.
		'''
		n1 = len(progenitor1)
		n2 = len(progenitor2)
		
		if n1 != n2:
			msg = 'Tamanho ind1 {0} diferente de ind2 {1}'.format(n1,n2)
			raise NoCompatibleIndividualSize(msg)
			
		order = list(range(n1))
		shuffle(order)
		
		ponto = randint(1, n1-1)
		desc1 = progenitor1.copy()
		desc2 = progenitor2.copy()
		
		desc1[:] = desc1[order]
		desc2[:] = desc2[order]
		
		desc1[ponto:] = desc2[ponto:]
		desc2[ponto:] = desc1[ponto:]
		
		tmp1 = desc1.copy()
		tmp2 = desc2.copy()	
		
		for i, j in enumerate(order):
			desc1[j] = tmp1[i]
			desc2[j] = tmp2[i]
			
		return desc1, desc2
		
		
		
		
