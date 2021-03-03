#!/usr/bin/python3
'''
Meant to solve problem in superconducting qubits with shifted frequency
'''

import jax
from jax import jit, vmap, grad
from jax import random
import numpy as np
import jax.numpy as jnp
from jax import vjp

from jax.experimental import ode, optimizers
from jax.experimental.ode import odeint

from jax.config import config  # Force Jax use float64 as default Float dtype
config.update("jax_enable_x64", True)

key = random.PRNGKey(42)

N = 1 # single qubit
N1 = 10 # parameter space size

sz = jnp.array([[1,0],[0,-1]],dtype=jnp.float32)

sx= jnp.array([[0,1],[1,0]],dtype=jnp.float32)

@jit
def A(t,p,t1):
    '''
    Control field, 
    '''
    w = jnp.pi/t1
    ft = jnp.array([jnp.sin(w*(i+1)*t) for i in range(N1)])
    return p@ft

@jit
def loss(t1,flat_p, omega, U_T):
    '''
    define the loss function, which is a pure function
    '''
    t_set = jnp.linspace(0., t1, 5)

    D, _, = jnp.shape(U_T)
    U_0 = jnp.eye(D,dtype=jnp.complex128)

    def func(y, t, *args):
        t1, omega, flat_p, = args

        return -1.0j*( omega* sz + A(t,flat_p,t1) * sx )@y
        # return -1.0j*( omega* sz)@y

    res = odeint(func, U_0, t_set, t1, omega, flat_p, rtol=1.4e-10, atol=1.4e-10)
    
    U_F = res[-1, :,:]
    return (1 - jnp.abs(jnp.trace(U_T.conj().T@U_F)/D)**2)


def GateOptimize(U_F, omega, t1, init_param, num_step=300, learning_rate=1.0):
    '''
    Get the best possible parameter
    psi_i: initial wave function
    psi_f: final wave function
    init_param: initial parameters
    '''
    opt_init, opt_update, get_params = optimizers.adam(
        learning_rate)  # Use adam optimizer
    loss_list = []

    def unpack(x):
                # t , args
        return x[0], x[1:]

    def step_fun(step, opt_state, U_F):
        aug_params = get_params(opt_state)
        t1, flat_params = unpack(aug_params)
        value, grads = jax.value_and_grad(
            loss, (0, 1))(t1, flat_params, omega, U_F) # use jax grad
        
        g_t, g_p = grads
        aug_grad = jnp.concatenate([jnp.array([g_t]), g_p])
        opt_state = opt_update(step, aug_grad, opt_state)
        return value, opt_state

    aug_params = jnp.concatenate([jnp.array([t1]), init_param])
    opt_state = opt_init(aug_params)

    # optimize
    for step in range(num_step):
        value, opt_state = step_fun(step, opt_state, U_F)
        loss_list.append(value)
        print('step {0} : loss is {1}'.format(
            step, value), end="\r", flush=True)

    print('final loss = ', value, flush=True)
    return loss_list, unpack(get_params(opt_state))

if __name__=='__main__':
    omega0 = 1.
    p = random.normal(key,shape=(N1,))
    t1 = 1.
    GateOptimize(sx,omega0, t1, p, num_step=300, learning_rate=1.2)
