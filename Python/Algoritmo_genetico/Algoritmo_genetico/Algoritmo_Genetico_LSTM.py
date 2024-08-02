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
from pygenec.evolucao_LSTM import Evolucao

from LSTM_project import LSTMgen

def fun(batch, sequence, hidden, nLayers, lr): #função que descreve a variação de Z em função de X e Y
    output = LSTMgen.mainLSTM(batch, sequence, hidden, nLayers, lr, tamanho_populacao=50,num_epochs=10)
    return output

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

def normaliza(populacao, genes_totais, genesPorFeatures):
    firstPass=True
    arrays={}
    count=0
    for i in range(genes_totais):
        if (i+1) % genesPorFeatures == 0:
            indi = f'indi{count}'
            if firstPass:
                conteudo = populacao[:,:i]
                arrays[indi] = conteudo
                firstPass=False
            elif not firstPass:
                conteudo = populacao[:,value:i]
                arrays[indi] = conteudo
            count+=1
            value=i
    batch=arrays['indi0']
    sequence=arrays['indi1']
    hidden=arrays['indi2']
    nLayers=arrays['indi3']
    lr=arrays['indi4']

    batch = np.asarray(batch)
    sequence = np.asarray(sequence)
    hidden = np.asarray(hidden)
    nLayers = np.asarray(nLayers)
    lr = np.asarray(lr)

    maiorbinBatch=2.0**batch.shape[1] - 1.0
    maiorbinSequence=2.0**sequence.shape[1] - 1.0
    maiorbinHidden=2.0**hidden.shape[1] - 1.0
    maiorbinLayers=2.0**nLayers.shape[1] - 1.0
    maiorbinLr=2.0**lr.shape[1] - 1.0

    minLr = 0.9            #    batch_size = 12  # Tamanho das sequências analisadas
    maxLr = 0.0001         #    sequence_Length = 3 # Sequência de dados de entrada
    minBatch = 3           #    input_size = 1  # Número de características dos dados de entrada
    maxBatch = 90          #     hidden_size = 50  # Tamanho do hidden state
    minSequence = 3        #    num_layers = 2  # Número de camadas LSTM
    maxSequence = 90       #    output_size = 1  # Número de características de saída
    minHidden = 10         #    num_epochs = 200  # Número de épocas de treinamento
    maxHidden = 100        #    learning_rate = 0.001  # Taxa de aprendizado
    minNlayers = 2
    maxNlayers = 10

    constLr = (maxLr - minLr) / maiorbinLr
    constBatch = (maxBatch - minBatch)  / maiorbinBatch
    constSequence = (maxSequence - minSequence) / maiorbinSequence
    constHidden = (maxHidden - minHidden) / maiorbinHidden
    constLayers = (maxNlayers - minNlayers) / maiorbinLayers

    lr = minBatch + constLr * bin(lr)

    batch1 = minBatch + constBatch * bin(batch)
    batch=[[0]*batch1.shape[1] for _ in range(batch1.shape[0])]
    batch = np.asarray(batch)
    for i in range(batch1.shape[0]):
        for j in range(batch1.shape[1]):
            batch[i,j] = int(batch1[i,j])

    sequence1 = minSequence + constSequence * bin(sequence)
    sequence=[[0]*sequence1.shape[1] for _ in range(sequence1.shape[0])]
    sequence = np.asarray(sequence)
    for i in range(sequence1.shape[0]):
        for j in range(sequence1.shape[1]):
            sequence[i,j] = int(sequence1[i,j])

    hidden1 = minHidden + constHidden * bin(hidden)
    hidden=[[0]*hidden1.shape[1] for _ in range(hidden1.shape[0])]
    hidden = np.asarray(hidden)
    for i in range(hidden1.shape[0]):
        for j in range(hidden1.shape[1]):
            hidden[i,j] = int(hidden1[i,j])

    nLayers1 = minNlayers + constLayers * bin(nLayers)
    nLayers=[[0]*nLayers1.shape[1] for _ in range(nLayers1.shape[0])]
    nLayers = np.asarray(nLayers)
    for i in range(nLayers1.shape[0]):
        for j in range(nLayers1.shape[1]):
            nLayers[i,j] = int(nLayers1[i,j])

    batch = batch[:,(batch.shape[1]-3)]
    sequence = sequence[:,(sequence.shape[1]-3)]
    hidden = hidden[:,(hidden.shape[1]-3)]
    nLayers = nLayers[:,(nLayers.shape[1]-3)]
    lr = lr[:,(lr.shape[1]-2)]

    return batch, sequence, hidden, nLayers, lr

def avaliacao(populacao, m='max'): # retorna o valor resultante dos indivíduos X e Y
    if m=='max':
        batch, sequence, hidden, nLayers, lr = normaliza(populacao, genes_totais, genesPorFeatures)
        tmp = fun(batch, sequence, hidden, nLayers, lr)
        return tmp
    elif m=='min':
        batch, sequence, hidden, nLayers, lr = normaliza(populacao, genes_totais, genesPorFeatures)
        tmp = -fun(batch, sequence, hidden, nLayers, lr)
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

for epoch in range(epochs):
    evolucao.evoluir()
#    batch, sequence, hidden, nLayers, lr = normaliza(populacao.populacao, genes_totais, genesPorFeatures)
#    output = fun(batch, sequence, hidden, nLayers, lr, tamanho_populacao)
#    print(f'MAE:{output}')
'''
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
'''
