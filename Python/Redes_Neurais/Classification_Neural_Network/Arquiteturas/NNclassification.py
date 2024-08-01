import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd
import pickle
from ucimlrepo import fetch_ucirepo

#modelo
class NnModel():
  def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons: int = 10, output_neurons: int = 2):
    np.random.seed(8)
    self.x = x
    self.y = y
    self.hidden_neurons = hidden_neurons
    self.output_neurons = output_neurons
    self.input_neurons = self.x.shape[1]

    # Incialização de pesos e bias
    self.w1 = np.random.randn(self.input_neurons, self.hidden_neurons) / np.sqrt(self.input_neurons)
    self.b1 = np.zeros((1, self.hidden_neurons))
    self.w2 = np.random.randn(self.hidden_neurons, self.output_neurons) / np.sqrt(self.hidden_neurons)
    self.b2 = np.zeros((1, self.output_neurons))
    self.model_dict = {'w1': self.w1, 'w2': self.w2, 'b1': self.b1, 'b2': self.b2}
    self.f1 = 0
    self.z1 = 0

  def foward(self, x: np.ndarray) -> np.ndarray:
    # Equação 1 da reta
    self.z1 = x.dot(self.w1) + self.b1                                          #(500,2) * (2,10) = (500,10)

    # Função 1 de ativação
    self.f1 = np.tanh(self.z1)                                                  #(500,10)

    # Equação 2 da reta
    z2 = self.f1.dot(self.w2) + self.b2                                         #(500,10) * (10,2) = (500,2)

    # Softmax (probabilidade)
    exp_values = np.exp(z2)
    softmax = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
    return softmax                                                              #(500,2)

  def loss(self, softmax):
    #Cross Entropy
    predictions = np.zeros(self.y.shape[0])
    log_probs = np.zeros(self.y.shape[0])

    for i, correct_index in enumerate(self.y):
      predicted = softmax[i][correct_index]
      predictions[i] = predicted
      log_probs[i] = -np.log(predicted)

    erro = np.sum(log_probs) / self.y.shape[0]
    return erro

  def backpropagation(self, softmax: np.ndarray, learning_rate: float) -> None:
    # Derivadas para atualizações de pesos e bias
    delta2 = np.copy(softmax)                                                   #(500.2)
    delta2[range(self.x.shape[0]), self.y] -= 1                                 #(500,)
    dw2 = (self.f1.T).dot(delta2)                                               #(10,500)*(500,) = (10,)
    db2 = np.sum(delta2, axis = 0, keepdims = True)
    delta1 = delta2.dot(self.w2.T)*(1-np.power(np.tanh(self.z1),2))
    dw1 = (self.x.T).dot(delta1)
    db1 = np.sum(delta1, axis = 0, keepdims = True)

    # Atulização dos pesos
    self.w1 += - learning_rate*dw1
    self.w2 += - learning_rate*dw2
    self.b1 += - learning_rate*db1
    self.b2 += - learning_rate*db2

  def fit(self, epochs: int, lr: float):

    for epoch in range(epochs):

      outputs = self.foward(self.x)
      loss = self.loss(outputs)
      self.backpropagation(outputs, lr)

      # Acurácia
      prediction = np.argmax(outputs, axis = 1)
      correct = (prediction == self.y).sum()
      accuracy = correct/self.y.shape[0]
      accuracy = accuracy*100

      if int((epoch+1) % (epochs/10)) == 0:
        print(f'Epoch: [{epoch + 1} / {epochs}]  Accuracy: {accuracy:.3f}%  Loss: {loss.item():.4f}')
        plt.scatter(self.x[:,0], self.x[:,1], c = prediction, s = 50, alpha = 0.05, cmap = 'cool')


    return prediction
