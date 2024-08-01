import autograd.numpy as np
#import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd
import pickle
from autograd import grad
from ucimlrepo import fetch_ucirepo

#plt.rcParams['figure.figsize'] = (10,6)
#plt.style.use('dark_background')
#modelo
class NnModel:
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons: int = 10, output_neurons: int = 2):
        np.random.seed(8)
        self.x = x
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.input_neurons = self.x.shape[1]

        # Inicialização de pesos e bias
        self.w1 = np.random.randn(self.input_neurons, self.hidden_neurons) / np.sqrt(self.input_neurons)
        self.b1 = np.zeros((1, self.hidden_neurons))
        self.w2 = np.random.randn(self.hidden_neurons, self.output_neurons) / np.sqrt(self.hidden_neurons)
        self.b2 = np.zeros((1, self.output_neurons))

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = np.dot(x, self.w1) + self.b1  # Use autograd.numpy para todas as operações
        f1 = np.tanh(z1)  # Use autograd.numpy para tanh

        z2 = np.dot(f1, self.w2) + self.b2  # Use autograd.numpy para operações de matriz/dot

        # Softmax (probabilidade)
        exp_values = np.exp(z2)  # Use autograd.numpy para operações exponenciais
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # autograd.numpy para soma
        return softmax

    def loss_function(self, params, x, y):
        # Desempacotando pesos e biases do parâmetro
        weights, biases = params
        w1, w2 = weights
        b1, b2 = biases

        # Calcular a saída usando o passo forward
        z1 = np.dot(x, w1) + b1  # autograd.numpy para dot
        f1 = np.tanh(z1)  # autograd.numpy para tanh
        z2 = np.dot(f1, w2) + b2  # autograd.numpy para dot
        softmax = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # autograd.numpy para operações

        # Cross Entropy
        log_probs = -np.log(softmax[range(y.shape[0]), y])
        return np.mean(log_probs)

    def fit(self, epochs: int, learning_rate: float):
        # Gradiente da função de perda para pesos e biases
        gradient_loss = grad(self.loss_function, argnum=0)  # autograd.numpy

        for epoch in range(epochs):
            # Criar lista de parâmetros para gradientes
            params = ((self.w1, self.w2), (self.b1, self.b2))  # Certifique-se de passar corretamente

            # Obter gradientes usando Autograd
            grads = gradient_loss(params, self.x, self.y)

            # Desempacotar gradientes
            dw1, dw2 = grads[0]
            db1, db2 = grads[1]

            # Atualizar pesos e biases
            self.w1 -= learning_rate * dw1
            self.w2 -= learning_rate * dw2
            self.b1 -= learning_rate * db1
            self.b2 -= learning_rate * db2

            # Calcular saída e perda
            outputs = self.forward(self.x)  # autograd.numpy
            loss = self.loss_function(params, self.x, self.y)  # autograd.numpy

            # Calcular precisão
            prediction = np.argmax(outputs, axis=1)  # autograd.numpy
            correct = (prediction == self.y).sum()  # autograd.numpy
            accuracy = correct / self.y.shape[0] * 100

            if (epoch + 1) % (epochs / 10) == 0:
                print(f'Epoch: [{epoch + 1} / {epochs}]  Accuracy: {accuracy:.3f}%  Loss: {loss:.4f}')
                #plt.scatter(self.x[:,0], self.x[:,1], c = prediction, s = 50, alpha = 0.05, cmap = 'cool')

        return prediction
