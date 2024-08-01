import numpy as np
#import matplotlib.pyplot as plt
from Arquiteturas.NNclassification_autograd import NnModel
from dataset import xy
import time

inicio = time.time()
#plt.rcParams['figure.figsize'] = (10,6)
#plt.style.use('dark_background')

x,y = xy()

hidden_neurons = 10
output_neurons = 30
learning_rate = 1
epochs = 100

modelo = NnModel(x,y,hidden_neurons=hidden_neurons,output_neurons=output_neurons)
#soft = modelo.foward(x)
#max = soft[range(x.shape[0]),y]
#max.shape
#modelo.loss(soft)

result = modelo.fit(epochs, learning_rate)
fim = time.time()
print(f'tempo em execução: {fim-inicio:.2f} segundos')
