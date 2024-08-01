import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNAgent(nn.Module):
	def __init__(self, state_shape, n_actions, epsilon=0):
		super().__init__()
		self.epsilon = epsilon
		self.n_actions = n_actions
		self.state_shape = state_shape
		state_dim = state_shape[0]
		# a simple NN with state_dim as input vector (inout is state s)
		# and self.n_actions as output vector of logits of q(s, a)
		self.network = nn.Sequential()
		self.network.add_module('layer1', nn.Linear(state_dim, 192))
		self.network.add_module('relu1', nn.ReLU())
		self.network.add_module('layer2', nn.Linear(192, 256))
		self.network.add_module('relu2', nn.ReLU())
		self.network.add_module('layer3', nn.Linear(256, 64))
		self.network.add_module('relu3', nn.ReLU())
		self.network.add_module('layer4', nn.Linear(64, n_actions))
		#
		self.parameters = self.network.parameters
	def forward(self, state_t):
		# pass the state at time t through the network to get Q(s,a)
		qvalues = self.network(state_t)
		return qvalues
	def get_qvalues(self, states):
		# input is an array of states in numpy and output is Qvals as numpy array
		states = torch.tensor(states, device=device, dtype=torch.float32)
		qvalues = self.forward(states)
		return qvalues.data.cpu().numpy()
	def sample_actions(self, qvalues):
		# sample actions from a batch of q_values using epsilon greedy policy
		epsilon = self.epsilon
		batch_size, n_actions = qvalues.shape
		random_actions = np.random.choice(n_actions, size=batch_size)
		best_actions = qvalues.argmax(axis=-1)
		should_explore = np.random.choice(
		[0, 1], batch_size, p=[1-epsilon, epsilon])
		return np.where(should_explore, random_actions, best_actions)
