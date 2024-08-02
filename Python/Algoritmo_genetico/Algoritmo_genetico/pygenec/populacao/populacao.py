from numpy.random import randint
from numpy import argsort, unique
import numpy as np

class Populacao:
    def __init__(self, avaliacao, genes_totais, tamanho_populacao):
        self.avaliacao = avaliacao
        self.genes_totais = genes_totais
        self.tamanho_populacao = tamanho_populacao

    def gerar_populacao(self):
        self.populacao = randint(0, 2, size=(self.tamanho_populacao, self.genes_totais), dtype='b')

    def avaliar(self, m='max'):
        u, indices = unique(self.populacao  , return_inverse=True, axis=0)
        #print(f'u\n{u}\nindices\n{indices}')
        if m=='max':
            valores = self.avaliacao(u)
        elif m=='min':
            valores = self.avaliacao(u,'min')
        #print(f'valores\n{valores}')

        valores = valores[indices]
        #print(f'valores[indices]\n{valores[indices]}')

        ind1 = argsort(valores)
        #print(f'ind1\n{ind1}')

        ind2 = ind1+ind1.shape[1]
        #print(f'ind2\n{ind2}\n{type(ind2)}')

        ind = np.concatenate((ind1,ind2), axis=1)
        #print(f'ind\n{ind}\n{type(ind)}')

        popu = self.populacao
        #print(f'popupalação\n{popu}\n')

        self.populacao[:] = np.take_along_axis(self.populacao, ind, axis=1)
        #print(f'população[ind]\n{self.populacao}')

        return np.take_along_axis(valores, ind1, axis=1)

