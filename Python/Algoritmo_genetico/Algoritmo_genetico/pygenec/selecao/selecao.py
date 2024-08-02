from numpy import array
import numpy as np

class Selecao:

	def __init__(self, populacao):
		self.populacao = populacao
		self.contador = 0
		
	def ind_rand(x):
		return np.random.uniform(0, x.shape[0])	
		
	def selecionar(self, fitness):
		"""
		Retorna a lista de índices do vetor populacao dos individuos selecionados
		"""
		raise NotImplementedError('A ser implementado')
		
	def selecao(self, n, fitness=None):
		"""
		Retorna a população de tamanho n, selecionada via metodo selecionar
		"""
		''' progenitores são os melhores indices'''
		progenitores = array([self.selecionar(fitness) for _ in range(n)])
		print(f'progenitores\n{progenitores}\nresult\n{self.populacao.populacao[progenitores]}\n')
		return self.populacao.populacao[progenitores]
			

