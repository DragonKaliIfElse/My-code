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
		# pass the state at time t through the newrok to get Q(s,a)
		qvalues = self.network(state_t)
		return qvalues

	def get_qvalues(self, states):
	# input is an array of states in numpy and outout is Qvals as numpy array
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

class ReplayBuffer:
	 def __init__(self, size):
		 self.size = size #max number of items in buffer
		 self.buffer =[] #array to hold samples
		 self.next_id = 0

	 def __len__(self):
		 return len(self.buffer)

	 def add(self, state, action, reward, next_state, done):
		 item = (state, action, reward, next_state, done)
		 if len(self.buffer) < self.size:
			 self.buffer.append(item)
		 else:
			 self.buffer[self.next_id] = item
		 self.next_id = (self.next_id + 1) % self.size

	 def sample(self, batch_size):
		 idxs = np.random.choice(len(self.buffer), batch_size)
		 samples = [self.buffer[i] for i in idxs]
		 states, actions, rewards, next_states, done_flags = list(zip(*samples))
		 return np.array(states),
				 np.array(actions),
				 np.array(rewards),
				 np.array(next_states),
				 np.array(done_flags)

class PlayAndRecord:
	def __init__(self, start_state, agent, env, exp_replay, n_steps=1):
		self.start_state = start_state
		self.agent = agent
		self.env = env
		self.exp_replay = exp_replay
		self.n_steps = n_steps

	def play_and_record():
		 s = self.start_state
		 sum_rewards = 0
		 # Play the game for n_steps and record transitions in buffer
		 for _ in range(self.n_steps):
		 qvalues = self.agent.get_qvalues([s])
		 a = self.agent.sample_actions(qvalues)[0]
		 next_s, r, done, _ = self.env.step(a)
		 sum_rewards += r
		 self.exp_replay.add(s, a, r, next_s, done)
		 if done:
		 s = self.env.reset()
		 else:
		 s = next_s
		 return sum_rewards, s

class
