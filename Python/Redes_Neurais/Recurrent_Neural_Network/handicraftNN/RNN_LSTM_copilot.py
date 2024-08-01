import numpy as np

class RnnLSTM():
	def __init__(self, x:np.ndarray, y:np.ndarray, hidden_neurons:int=20, output_neurons:int=1, tipo:string = 'Regressão'):
        self.x = x
        self.y = y
        self.input_size = self.x.shape[1]
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        
        # Inicialização dos pesos e vieses para as portas de entrada, esquecimento, célula e saída
        self.Wf = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wi = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wc = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wo = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
		self.Wf2 = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wi2 = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wc2 = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.Wo2 = np.random.randn(self.hidden_neurons, self.input_size + self.hidden_neurons) #Px(p+d)
        self.bf = np.zeros((self.hidden_neurons, self.ouput_neurons)) #PxO
        self.bi = np.zeros((self.hidden_neurons, self.ouput_neurons)) #PxO
        self.bc = np.zeros((self.hidden_neurons, self.ouput_neurons)) #PxO
        self.bo = np.zeros((self.hidden_neurons, self.ouput_neurons)) #PxO
        
        # Inicialização do estado da célula e do estado oculto
        self.c = np.zeros((self.hidden_neurons, self.output_neurons))
        self.h = np.zeros((self.hidden_neurons, self.output_neurons))
        
        # Tipo de RNN
        self.tipo = tipo
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def d_sigomid(self, x):
    	return np.log(np.exp(x) + 1)
    	
    def softmax(self, x):
    	exp_values = np.exp(x)
		softmax = exp_values / np.sum(exp_values, axis = 0, keepdims = True)
		return softmax
    
    def forward_t(self, x: np.ndarray):
        # Concatenação da entrada e do estado oculto anterior
        combined = np.concatenate((x, self.h), axis=0) #(d+p)xO
        '''
        combined_Wf = np.concatenate((self.Wf, self.Wf2), axis=0) #2Px(d+p)
        combined_Wi = np.concatenate((self.Wi, self.Wi2), axis=0) #2Px(d+p)
        combined_Wo = np.concatenate((self.Wo, self.Wo2), axis=0) #2Px(d+p)
        combined_Wc = np.concatenate((self.Wc, self.Wc2), axis=0) #2Px(d+p)
        '''
        self.Xf = combines_Wf.dot(combined) #2PxO
        self.Xi = combines_Wi.dot(combined) #2PxO
        self.Xc = combines_Wc.dot(combined) #2PxO
        self.Xo = combines_Wo.dot(combined) #2PxO
        
        # Porta de esquecimento
        self.ft = self.sigmoid(self.Xf + self.bf)
        # Porta de entrada
        self.it = self.sigmoid(self.Xi + self.bi)
        # Atualização do estado da célula
        self.ct_ = np.tanh(self.Xc + self.bc)
        # Atualização final do estado da célula
        self.c = self.ft * self.c + self.it * self.ct_
        
        # Porta de saída
        self.ot = self.sigmoid(self.Xo + self.bo)
        # Estado oculto
        self.h = self.ot * np.tanh(self.c)
        
        # Probabilidade para Classificação
        if self.tipo == 'Classificação': 
        	hidden_state = self.softmax(self.h)
		elif self.tipo == 'Regressão':
			hidden_state = np.sum(self.h, axis=0, keppdims=True)
			
        return hidden_state
    
	def loss(self, hidden_state):
					
		if self.tipo == 'Classificação': #Cross entroopy
		
				log_probs = np.zeros(self.y.shape[0])

				for t, correct_class_ in enumerate(self.y):
				  predicted = hidden_state[t][correct_class_]	
				  log_probs[t] = -np.log(predicted)

				erro = np.sum(log_probs)
			
		elif self.tipo == 'Regressão':
			
			
		return erro
			
	def BTT(self, hidden_state, learning_rate): #Backpropagation Trough Time
		if self.tipo == 'Classificação':	
			delta = np.copy(hidden_state)
			delta[range(self.x.shape[0]), self.y] -= 1
			dWo = sigmoid(combined_Xo).dot(delta)
			dBo = np.sum(delta, axis = 0, keepdims = True)
			
		
	def fit(self):
	
		
		
		
