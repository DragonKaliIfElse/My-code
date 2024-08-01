import numpy as np

class RnnLSTM():
	def __init__(self, x: np.ndarray, y: np.ndarray, output_neurons: int =1):
		self.x=x
		self.y=y
		self.output_neurons=output_neurons

		self.w1 = np.random.rand()
		self.w2 = np.random.rand()
		self.w3 = np.random.rand()
		self.w4 = np.random.rand()
		self.w5 = np.random.rand()
		self.w6 = np.random.rand()
		self.w7 = np.random.rand()
		self.w8 = np.random.rand()
		self.b1 = 0
		self.b2 = 0
		self.b3 = 0
		self.b4 = 0
		self.LTM = 0
		self.STM = 0

	def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigomid(self, x):
    	return np.log(np.exp(x) + 1)

    def softmax(self, x):
    	exp_values = np.exp(x)
		softmax = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
		return softmax

	def forward(self, input_data): #Long Short Term Memory algorithm
		self.x1 = (input_data * self.w1) + (self.STM * self.w2) + self.b1
		self.x2 = (input_data * self.w3) + (self.STM * self.w4) + self.b2
		self.x3 = (input_data * self.w5) + (self.STM * self.w6) + self.b3
		self.x4 = (input_data * self.w7) + (self.STM * self.w8) + self.b4

		LTTR = self.sigmoid(self.x1)			#Long Term to Remember
		PMTR = self.sigmoid(self.x2)			#Potencial Memory to Remember
		PMR2 = self.sigmoid(self.x3)			#Potencial Memory to Remember
		PLTM = np.tanh(self.x4)				#Potencial Long Term Memory

		FG = LTTR * self.LTM			#Forget Gate
		self.LTM = PMTR * PLTM + FG		#Input Gate

		PSTM = np.tanh(self.LTM)		#Potencial Short Term Memory

		self.STM = PMR2 * PSTM			#Output Gate
		output = softmax(self.STM)

		return output

	def loss(self):

	def backpropagation(self):

	def fit(self):

