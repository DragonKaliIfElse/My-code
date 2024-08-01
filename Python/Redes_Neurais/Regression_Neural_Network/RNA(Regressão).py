import numpy as np
 

class NnModel():
  def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons1: int = 10, hidden_neurons2: int = 10, output_neurons: int = 1):
    np.random.seed(8)

    self.x = x
    self.y = y
    self.hidden_neurons1 = hidden_neurons1
    self.hidden_neurons2 = hidden_neurons2
    self.output_neurons = output_neurons
    self.input_neurons = self.x.shape[1]

    # Incialização de pesos e bias
    self.w1 = np.random.randn(self.input_neurons, self.hidden_neurons1) / np.sqrt(self.input_neurons)
    self.b1 = np.zeros((1,self.hidden_neurons1))
    self.w2 = np.random.randn(self.hidden_neurons1, self.hidden_neurons2) / np.sqrt(self.hidden_neurons1)
    self.b2 = np.zeros((1, self.hidden_neurons2))
    self.w3 = np.random.randn(self.hidden_neurons2, self.output_neurons) / np.sqrt(self.hidden_neurons2)
    self.b3 = np.zeros((1, self.output_neurons))
    self.model_dict = {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'b1': self.b1, 'b2': self.b2,'b3': self.b3}
    self.f1 = 0
    self.f2 = 0
    self.z1 = 0
    self.z2 = 0

  def foward(self, x: np.ndarray) -> np.ndarray:
    # Equação 1 da reta
    self.z1 = x.dot(self.w1) + self.b1                                          #(768,3)*(3,10)

    # Função 1 de ativação: Tangente hiperbólica
    self.f1 = np.tanh(self.z1)                                                  #(768,10)

    # Equação 2 da reta
    self.z2 = self.f1.dot(self.w2) + self.b2                                    #(768,10)*(10,10)

    # Função 2 de ativação: Softmax (probabilidade)
    exp_values = np.exp(self.z2)
    softmax = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
    self.f2 = softmax                                                           #(768,10)

    # Equação 3 da reta
    z3 = self.f2.dot(self.w3) + self.b3                                         #(768,10)*(10,1)

    # Função 3 de ativação: Linear
    linear = z3
    return linear                                                                #(768,1)

  def loss(self, linear):
    # Mean Squad Error (MSE)

    dif_square = (self.y - linear)**2
    self.somatorio = (self.y.shape[0])
    MSE = (dif_square)/self.somatorio
    return MSE

  def backpropagation(self, linear: np.ndarray, learning_rate: float) -> None:
    # Derivadas para atualizações de pesos e bias
    # de/dz3 * dz3/dw3 * dw3/df2 * df2/dz2 * dz2/dw2 * dw2/df1 * df1/dz1 * dz1/dw1 = de/dw1 (função de perda em função do peso 1)

    # delta3 = de/dz3 * dz3/dw3
    linear1 = np.copy(linear)                                                   #(768,1)
    d_dif_square = (-2)*(self.y - linear1)                                      #(768,1)                                    
    MSE_d = (d_dif_square)/self.somatorio                                       #(1)
    delta3 = linear1*MSE_d                                                      #(768,1)
    dw3 = (self.f2.T).dot(delta3)                                               #(10,768)*(768,1)
    db3 = np.sum(delta3, axis = 0, keepdims = True)

    # delta2 = dw3.reshape(-1, 1)/df2 * df2/dz2 * dz2/dw2
    d_softmax = self.f2*(1 - self.f2)
    delta31 = delta3.reshape(-1, 1)
    delta2 = delta31.dot(self.w3.T) * d_softmax
    dw2 = (self.f1.T).dot(delta2)
    db2 = np.sum(delta2, axis = 0, keepdims = True)

    # delta1 = dw2/df1 * df1/dz1 * dz1/dw1
    # sec² + tan² = 1 . sec² = 1 - tan²
    d_tanh = 1 - np.square(self.f1)
    delta1 = delta2.dot(self.w2.T)*d_tanh
    dw1 = (self.x.T).dot(delta1)
    db1 = np.sum(delta1, axis = 0, keepdims = True)

    # Atulização dos pesos
    self.w1 += - learning_rate*dw1
    self.w2 += - learning_rate*dw2
    self.w3 += - learning_rate*dw3
    self.b1 += - learning_rate*db1
    self.b2 += - learning_rate*db2
    self.b3 += - learning_rate*db3

  def fit(self, epochs: int, lr: float):

    for epoch in range(epochs):

      outputs = self.foward(self.x)
      loss = self.loss(outputs)
      self.backpropagation(outputs, lr)

      # Acurácia
      prediction = np.argmax(outputs, axis = 1)
      correct = (prediction == self.y).sum()
      accuracy = correct/self.y.shape[0]

      if int((epoch+1) % (epochs/10)) == 0:
        print(f'Epoch: [{epoch + 1} / {epochs}]  Accuracy: {accuracy:.3f}  Loss: {loss.item():.4f}')
        plt.scatter(x[:,0], x[:,1], c = prediction, s = 50, alpha = 0.05, cmap = 'cool')


    return prediction

