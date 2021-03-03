#/usr/bin/python3
from scipy.optimize import minimize

from jax.experimental.ode import odeint
from jax.experimental import ode, optimizers
import numpy as np
from jax import hessian, jacrev
from Xgate import loss
from Xgate import GateOptimize
import jax.numpy as jnp
import jax.random as random
import jax
from jax import jit, vmap, grad


key = random.PRNGKey(42)

N = 1  # single qubit
N1 = 10  # parameter space size

sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)

sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)

def Hessian_eigen(p_final):
    '''
    Calculate main eigen vectors of loss Hessian matrix around optmized parameter
    '''
    loss_p = lambda p: loss(p_final[0], p,omega0,sx)
    Hess = jacrev(jacrev(loss_p))(p_final[1])

    w, v = jnp.linalg.eig(Hess)

    arglist = jnp.argsort(w)[-3:]
    main_eigen = w[arglist]
    main_vec = v[:, arglist]

    return main_eigen, main_vec

@jit
def A(t, p, t1):
    '''
    Control field, 
    '''
    w = jnp.pi/t1
    ft = jnp.array([jnp.sin(w*(i+1)*t) for i in range(N1)])
    return p@ft

@jit
def loss_raw(dp, t1, p0, omega, U_T):
    '''
    define the loss function of 
    Raw method
    , which is a pure function
    '''
    t_set = jnp.linspace(0., t1, 5)

    D, _, = jnp.shape(U_T)
    U_0 = jnp.eye(D, dtype=jnp.complex128)
    flat_p = p0 + dp

    def func(y, t, *args):
        t1, omega, flat_p, = args

        return -1.0j*(omega * sz + A(t, flat_p, t1) * sx)@y
        # return -1.0j*( omega* sz)@y

    res = odeint(func, U_0, t_set, t1, omega,
                 flat_p, rtol=1.4e-10, atol=1.4e-10)

    U_F = res[-1, :, :]
    return (1 - jnp.abs(jnp.trace(U_T.conj().T@U_F)/D)**2)


@jit
def loss_vec(dp, main_vec, t1, p0, omega, U_T):
    '''
    define the loss function of 
    main eigen method
    , which is a pure function
    '''
    t_set = jnp.linspace(0., t1, 5)

    D, _, = jnp.shape(U_T)
    U_0 = jnp.eye(D, dtype=jnp.complex128)

    flat_p = p0 + main_vec@dp

    def func(y, t, *args):
        t1, omega, flat_p, = args

        return -1.0j*(omega * sz + A(t, flat_p, t1) * sx)@y
        # return -1.0j*( omega* sz)@y

    res = odeint(func, U_0, t_set, t1, omega,
                 flat_p, rtol=1.4e-10, atol=1.4e-10)

    U_F = res[-1, :, :]
    return (1 - jnp.abs(jnp.trace(U_T.conj().T@U_F)/D)**2)


# def dloss_vec(dp, dw):
#     dp0 = jnp.zeros(shape=(3,))
#     return loss_vec(dp, t1_final, p0_final, omega0, sx) - loss_vec(dp0, t1_final, p0_final, omega, sx)
# 
# def dloss_raw(dp, dw):
    # dp0 = jnp.zeros(shape=(N1,))
    # return loss_raw(dp, t1_final, p0_final, omega0, sx) - loss_raw(dp0, t1_final, p0_final, omega, sx)


if __name__ == '__main__':
    omega0 = 1.
    p = random.normal(key, shape=(N1,))
    t0 = 1.
    gate_loss, p_final = GateOptimize(
        sx, omega0, t0, p, num_step=300, learning_rate=1.2)

    _ ,vector_mat = Hessian_eigen(p_final)
    # Now try different optimization method
    dw = 0.03

    t1_final = p_final[0]
    p0_final = p_final[1]

    print('starting optimize')

    res_vec = minimize(loss_vec,jnp.zeros(3,dtype=jnp.float32),
        args=(vector_mat,t1_final,p0_final,omega0+dw,sx),method='Nelder-Mead')
    
    print('result for hessian efficient method')
    print(res_vec.nit, res_vec.success)
    res_raw = minimize(loss_raw, jnp.zeros(10, dtype=jnp.float32),
                       args=(t1_final, p0_final, omega0+dw, sx), method='Nelder-Mead')
    
    print('result for raw Nelder')
    print(res_raw.nit, res_raw.success)
