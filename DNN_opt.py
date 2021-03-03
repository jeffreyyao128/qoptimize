#!~/miniconda3/bin/python
'''
Interprete control pulse using DNN
'''
import jax.numpy as jnp
import jax.random as random
import jax
from jax import jit, vmap, grad
key = random.PRNGKey(42)

from matplotlib import pyplot as plt

from typing import Any, Callable, Sequence, Optional
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging
# config.update("jax_enable_x64", True)# Enable complex128

key = random.PRNGKey(42)

N = 1  # single qubit
N1 = 10  # parameter space size

sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)

sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)

class ExplicitMLP(nn.Module):
    '''
    MLP class
    '''
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

A = ExplicitMLP(features=[4,16,64,16,1]) # define 5 layers DNN

from jax.experimental import ode, optimizers
from jax.experimental.ode import odeint

@jit
def loss(t1, omega, params, U_T):
    '''
    define the loss function, which is a pure function
    params as pytree
    '''
    t_set = jnp.linspace(0., t1, 5)

    D, _, = jnp.shape(U_T)
    U_0 = jnp.eye(D,dtype=jnp.complex64)

    def func(y, t, *args):
        omega, params, = args

        return -1.0j*( omega* sz + A.apply(params,jnp.array([t,omega])) * sx )@y #Using DNN as control field
        # return -1.0j*( omega* sz)@y

    res = odeint(func, U_0, t_set, omega, params, rtol=1.4e-10, atol=1.4e-10)
    
    U_F = res[-1, :,:]
    return (1 - jnp.abs(jnp.trace(U_T.conj().T@U_F)/D)**2)

def GateOptimizeOmega(U_F, omega, t1, init_param, num_step=300, learning_rate=1.0):
    '''
    Get the optimized parameter

    init_param: initial parameters
    '''
    opt_init, opt_update, get_params = optimizers.adam(
        learning_rate)  # Use adam optimizer
    loss_list = []

    def step_fun(step, opt_state, U_F):
        aug_params = get_params(opt_state)
        t1, params = aug_params
        value, grads = jax.value_and_grad(
            loss, (0, 2))(t1, omega, params, U_F) # use jax grad
        
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    aug_params = (t1, init_param)
    opt_state = opt_init(aug_params)

    # optimize
    for step in range(num_step):
        value, opt_state = step_fun(step, opt_state, U_F)
        loss_list.append(value)
        print('step {0} : loss is {1}'.format(
            step, value), end="\r", flush=True)

    print('final loss = ', value, flush=True)
    return loss_list, get_params(opt_state)

if __name__=='__main__':
    # Initialize DNN for A
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (2,)) # dummy input

    params = A.init(key2, x)
    omega0 = 1.
    p0 = random.normal(key,shape=(N1*4+1,))
    t1 = 1.

    # print(loss(t1,omega0,params,sx))
    # y = A.apply(params, x)

    # A_t = lambda omega, t: A.apply(params,jnp.array([omega,t]))

    # A_t = vmap(A_t,(None,0),0)

    # t = jnp.linspace(0,1.,100)
    # A_array = A_t(1.,t)

    # plt.plot(t,A_array)
    # plt.show()
    loss_list, p_final=GateOptimizeOmega(sx,omega0, t1, params, num_step=300, learning_rate=1.8)