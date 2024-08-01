import jax.numpy as jnp
from jax import grad, jit, vmap
import jax

key = jax.random.PRNGKey(0)
class NNC_jax():
	def __init__(self,params):
		self.params = params
	
	def net(self,x):
		layer1 = jnp.dot(x,self.params['w1'])+self.params['b1']
		#função de ativação 1
		layer1 = jnp.maximum(0,layer1)
		ouput = jnp.dot(layer1,self.params['w2']) + self.params['b2']
		return ouput

	def loss_fn(self,x,y):
		predictions = self.net(x)
		return jnp.mean((predictions-y)**2)
