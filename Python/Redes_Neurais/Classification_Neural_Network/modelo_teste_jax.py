from Arquiteturas.NNClas_jax import NNC_jax
from Arquiteturas import NNClas_jax as NNC
from dataset import xy
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
import time

x_train,y_train = xy()

inicio = time.time()

key = jax.random.PRNGKey(0)

params = {'w1':jax.random.normal(key,shape=(4,3)),'b1':jnp.zeros(4),
		'w2':jax.random.normal(key,shape=(2,3)), 'b2':jnp.zeros(2)}
		
model = NNC_jax(params)

grad_loss = grad(model.loss_fn)

for epoch in range(100):
	grads = grad_loss(x_train,y_train)
	params = update_params(params, grads)

print(params)
fim = time.time()

print(f'tempo em execução: {fim-inicio:.2f} segundos')
