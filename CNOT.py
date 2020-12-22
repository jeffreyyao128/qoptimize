#/usr/bin/python3

from jax.test_util import check_grads
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import jax
from jax import jit, vmap, grad
from jax import random
import numpy as np
import jax.numpy as jnp
from jax import vjp

from jax.experimental import ode, optimizers
from jax.experimental.ode import odeint

from jax.config import config # Force Jax use float64 as default Float dtype
config.update("jax_enable_x64", True)

N = 2  # set number of qubits as a global parameter
N1 = 10  # fourier series size
key = random.PRNGKey(42)

U_cnot = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def n(i, x):
    '''
    number operator for n_i (x)
    '''
    return (x & 2**i)/2**i

def initial(key1=None):
    '''
    Return randomized parameters
    '''
    return random.normal(key1,shape=(N1*(N+1)+1,))

def unpackp(x):
    return jnp.array(x[:N1]), jnp.array([x[(i+1)*N1:(i+2)*N1] for i in range(N)]), jnp.array(x[-1], dtype=jnp.float32)


n0 = jnp.array([jnp.diag(jnp.array([n(i, x) for x in range(2**N)]))
                for i in range(N)])  # number operator matrix form n0[i] = n_i

def H_independent():
    '''
    Time independent part of Hamiltonian
    '''
    res = jnp.zeros((2**N, 2**N))
    for i in range(N):
        for j in range(N):
            if j <= i:
                continue
            params = 1/np.abs(i-j)**6
            res += params*(jnp.dot(n0[i], n0[j]))
    return res

H1 = H_independent()
# print(jnp.trace(H1))

def f(x, y, i): return 1 if y == x ^ 2**i else 0


H2 = sum([jnp.array([[f(x, y, i) for x in range(2**N)]
                     for y in range(2**N)]) for i in range(N)])  # checked

@jit
def Hmat(t,flat_p,t1):
    '''
    Using dense matrix to represent Hamiltonian
    '''
    w = 2*jnp.pi/t1
    u_omega, u_d, V = unpackp(flat_p)

    ft = jnp.array([jnp.sin(w*(i+1)*t) for i in range(N1)])

    omega = jnp.dot(u_omega, ft)
    delta = u_d@ft

    return V*H1 + 0.5 * omega * H2 - jnp.einsum('i,ijk->jk', delta, n0)

@jit
def loss(psi_init, t1,flat_p, psi0):
    '''
    define the loss function, which is a pure function
    '''
    t_set = jnp.linspace(0., t1, 5)

    def func(y, t, *args):
        t1, flat_p, = args
        return -1.0j*Hmat(t, flat_p, t1)@y
    
    res = odeint(func, psi_init, t_set, t1, flat_p, rtol=1.4e-10, atol=1.4e-10)
    psi_final = res[-1, :]
    
    return (1 - jnp.abs(jnp.dot(jnp.conjugate(psi_final), psi0))**2)

if __name__=='__main__':
    psi_i = jnp.array([0,0,0,1],dtype=jnp.complex128)
    psi_f = 1/jnp.sqrt(2)*jnp.array([0, 1, 1, 0], dtype=jnp.complex128)
    t1 = 1.
    flat_p = initial(key)

    num_step = 300
    learning_rate=1.0
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)  # Use adam optimizer
    loss_list = []
    def unpack(x):
                # t , args
        return x[0], x[1:]

    def step_fun(step, opt_state, psi, psi0):
        aug_params = get_params(opt_state)
        t1, flat_params = unpack(aug_params)
        value, grads = jax.value_and_grad(loss, (1, 2))(psi, t1, flat_params, psi0)
        g_t, g_p = grads
        aug_grad = jnp.concatenate([jnp.array([g_t]), g_p])
        opt_state = opt_update(step, aug_grad, opt_state)
        return value, opt_state

    aug_params = jnp.concatenate([jnp.array([t1]), flat_p])
    opt_state = opt_init(aug_params)

    # optimize
    for step in range(num_step):
        value, opt_state = step_fun(step, opt_state, psi_i, psi_f)
        loss_list.append(value)
        print('step {0} : loss is {1}'.format(
            step, value))
    
    plt.plot([x for x in range(num_step)], loss_list)
    plt.show()
    

