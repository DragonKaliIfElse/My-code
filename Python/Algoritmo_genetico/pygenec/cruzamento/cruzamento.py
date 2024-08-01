import numpy as np
from numpy import array
from numpy.random import randint, random

class NoCompatibleIndividualSize(Exception):
	pass

class Cruzamento:
	'''
	Classe abstrata representando o cruzamento.
	
	Entrada:
		tamanho_populacao - Tamanho final da população resultante
	'''
	
	def __init__(self,tamanho_populacao):
		self.tamanho_populacao = tamanho_populacao
		
	def cruzamento(self, progenitor1, progenitor2):
		raise NotImplementedError("A ser implementado")
		
	def descendentes(self, subpopulacao, pcruz):
		'''
		Retorna uma nova população de tamanho tamanho_populacao através do cruzamento.
		
		Entrada:
			subpopulacao - Vetor contendo indivíduos para serem selecionados para cruzamento.
		
		pcruz - probabilidade de cruzamento entre dois indivíduos selecionados.
		'''
		nova_populacao = []
		npop = len(subpopulacao)
		
		while len(nova_populacao) < self.tamanho_populacao:
			i = randint(0, npop -1)
			j = randint(0, npop -1)
			while j == i:
				j = randint(0, npop -1)
				
			cruza = random()
			if cruza < pcruz:
				desc1, desc2 = self.cruzamento(subpopulacao[i], subpopulacao[j])
				nova_populacao.append(desc1)
				if len(nova_populacao) < self.tamanho_populacao:
					nova_populacao.append(desc2)
					
		return array(nova_populacao)			
			
			
			
			
			
			
			
