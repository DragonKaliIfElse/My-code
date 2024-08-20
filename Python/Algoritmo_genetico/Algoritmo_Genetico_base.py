#!/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from numpy import mgrid
from numpy import exp, array
from numpy.random import randint
from matplotlib.animation import FuncAnimation

from pygenec.populacao.populacao import Populacao
from pygenec.selecao.roleta import Roleta
from pygenec.selecao.classificacao import Classificacao
from pygenec.selecao.torneio import Torneio
from pygenec.cruzamento.kpontos import Kpontos
from pygenec.cruzamento.umponto import UmPonto
from pygenec.cruzamento.embaralhamento import Embaralhamento
from pygenec.mutacao.duplatroca import DuplaTroca
from pygenec.mutacao.flip import Flip
from pygenec.mutacao.sequenciareversa import SequenciaReversa
from pygenec.evolucao import Evolucao

def fun(x,y): #função que descreve a variação de Z em função de X e Y
	tmp = 3*exp(-(y+1)**2 - x**2)*(x-1)**2 - exp(-(x+1)**2 - y**2/3) + exp(-x**2 - y**2) * (10*x**3 - 2*x + 10*y**5 )
	return tmp

def bin(x): #converte valores binarios em inteiros
	cnt=array([2**i for i in range(x.shape[1])])
	return array([cnt * x[i,:].sum() for i in range(x.shape[0])])

def xy(populacao): #normaliza o intervalo de busca da função para valores de interesse
	colunas = populacao.shape[1]
	meio = colunas // 2
	maiorbin = 2.0**meio - 1.0
	nmin = -3
	nmax = 3
	const = (nmax - nmin) / maiorbin
	x = nmin + const * bin(populacao[:, :meio])
	y = nmin + const * bin(populacao[:,meio: ])
	return x,y


def avaliacao(populacao, m='max'): # retorna o valor resultante dos indivíduos X e Y
	if m=='max':
		x,y = xy(populacao)
		tmp = fun(x,y)
		return tmp
	elif m=='min':
		x,y = xy(populacao)
		tmp = -fun(x,y)
		return tmp
def m_value(x,m):
	if m == 'min':
		m=np.min
	elif m=='max':
		m=np.max
	value = m(x,axis=1)
	value = value.reshape(-1,1)
	for i in range(x.shape[0]):
		x[i] = value[i]
	x = x[:,0]
	return x

def ind_result(y, x, m):
	if m == 'min':
		m=np.argmin
	elif m=='max':
		m=np.argmax
	ind = m(y,axis=1)
	ind = ind.reshape(-1,1)
	for i in range(x.shape[0]):
		x[i,:] = x[i,ind[i]]
	x = x[:,0]
	return x

def selecao(x, populacao, tamanho=10, m='max'):
	if x == "roleta":
		selecao = Roleta(populacao, m)
	elif x == "classificação":
		selecao = Classificacao(populacao, m)
	elif x == "torneio":
		selecao = Torneio(populacao, tamanho, m)
	else:
		print("Escolha um método de seleção: roleta, classificação, torneio\n")
	return selecao

def cruzamento(x, tamanho_populacao):
	if x == 'kpontos':
		cruzamento = Kpontos(tamanho_populacao)
	elif x == 'umponto':
		cruzamento = UmPonto(tamanho_populacao)
	elif x == 'embaralhamento':
		cruzamento = Embaralhamento(tamanho_populacao)
	else:
		print('Escolha um método de cruzamento: kpontos, umponto, embaralhamento\n')
	return cruzamento

def mutacao(x, pmut):
	if x=='flip':
		mutacao = Flip(pmut)
	elif x=='duplatroca':
		mutacao = DuplaTroca(pmut)
	elif x=='sequenciareversa':
		mutacao = SequenciaReversa(pmut)
	else:
		print('Ecolha um dos métodos de mutação: flip, duplatroca, sequenciareversa\n')
	return mutacao


features = 5
genesPorFeatures = 16
genes_totais = features*genesPorFeatures
tamanho_populacao = 50

populacao = Populacao(avaliacao, genes_totais, tamanho_populacao)
populacao.gerar_populacao()
#populacao.avaliar()

m = 'min'

selecao = selecao("torneio", populacao, tamanho=10,m=m)
cruzamento = cruzamento("kpontos", tamanho_populacao)
mutacao = mutacao('duplatroca', pmut=0.9)
evolucao = Evolucao(populacao, selecao, cruzamento, mutacao, m=m)

evolucao.nsele = 10
evolucao.pcruz = 0.5
epochs = 10000

fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(111, projection='3d')
X, Y = mgrid[-3:3:30j, -3:3:30j]
Z = fun(X,Y)
ax.plot_wireframe(X, Y, Z)

x, y = xy(populacao.populacao)

z = fun(x,y)
x = ind_result(z,x, m)
y = ind_result(z,y, m)
z = m_value(z, m)

graph = ax.scatter(x, y, z, s=40, c='red', marker='D')

def update(frame):
	evolucao.evoluir()
	x,y = xy(populacao.populacao)
	z = fun(x,y)
	x = ind_result(z,x, m)
	y = ind_result(z,y, m)
	z = m_value(z, m)
	graph._offsets3d = (x,y,z)
	print(f'x\n{x}\ny\n{y}\nz\n{z}\n')

ani = FuncAnimation(fig, update, frames=range(10000), repeat=False)
plt.show()
