#/usr/bin/python3

import matplotlib.pyplot as plt

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

from sklearn.decomposition import PCA

key = random.PRNGKey(42)

N = 1  # single qubit
N1 = 10  # parameter space size

sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)

sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)

omega0 = 1.

def Hessian_eigen(p_final,n_size=3):
    '''
    Calculate main eigen vectors of loss Hessian matrix around optmized parameter
    '''
    loss_p = lambda p: loss(p_final[0], p,omega0,sx)
    Hess = jacrev(jacrev(loss_p))(p_final[1])

    w, v = jnp.linalg.eig(Hess)

    arglist = jnp.argsort(w)[-n_size:]
    main_eigen = w[arglist]
    main_vec = v[:, arglist]

    return main_eigen, main_vec # shape = (N1, n_size)

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

def PCA_visual(p_final):
    '''
    Visualize matrix (N* M) as N points
    '''

    loss_p = lambda p: loss(p_final[0], p,omega0,sx)
    Hess = jacrev(jacrev(loss_p))(p_final[1])

    w, v = jnp.linalg.eig(Hess)
    v_real = v.real
    arglist = jnp.argsort(w)[-3:]

    pca = PCA(n_components=2) # scatter in 2d pannel
    reduced = pca.fit_transform(v_real[:,arglist].transpose())

    t = reduced.transpose()
    # t_main = t[:,arglist]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(t[0], t[1],marker='v',label='main')
    # ax1.scatter(t[0], t[1],marker='o',label='full')
    # ax1.scatter(t_main[0],t_main[1],marker='v',label='main')

    plt.legend(loc='upper left')
    plt.show()

def Plot_vector(p_final):
    '''
    p_final: parameters of control field
    Plot all eigenvectors of Hessian matrix
    '''
    loss_p = lambda p: loss(p_final[0], p,omega0,sx)
    Hess = jacrev(jacrev(loss_p))(p_final[1])

    w, v = jnp.linalg.eig(Hess)
    v_real = v.real
    arglist = jnp.argsort(w)[-3:]
    x = range(10)
    for i in range(10):
        if i in arglist:
            plt.plot(x, v_real[:,i],label=f"main vector {i}")
    
    plt.xlabel("index")
    plt.ylabel("$V_{ij}$")

    plt.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    plt.show()
    



if __name__ == '__main__':
    p = random.normal(key, shape=(N1,))
    t0 = 1.
    gate_loss, p_final = GateOptimize(
        sx, omega0, t0, p, num_step=300, learning_rate=1.2)

    _ ,vector_mat = Hessian_eigen(p_final)
    # Now try different optimization method
    dw = 0.03

    # print(vector_mat)
    
    t1_final = p_final[0]
    p0_final = p_final[1]

    Plot_vector(p_final)
    # print('starting optimization')

    # PCA_visual()
    # res_vec = minimize(loss_vec,jnp.zeros(3,dtype=jnp.float32),
    #     args=(vector_mat,t1_final,p0_final,omega0+dw,sx),method='Nelder-Mead',options={'return_all':True}) # Intermediate state results are stored in res_vec.allvecs

    # res_raw = minimize(loss_raw, jnp.zeros(10, dtype=jnp.float32),
    #                    args=(t1_final, p0_final, omega0+dw, sx), method='Nelder-Mead',options={'return_all':True})
    # print(res_vec.allvecs.__len__)

    # res_single = [minimize(loss_vec,jnp.zeros(1,dtype=jnp.float32),
    #     args=(vector_mat[:,i].reshape(N1,1),t1_final,p0_final,omega0+dw,sx),
    #     method='Nelder-Mead',options={'return_all':True}) for i in range(3)]

    # res_double = minimize(loss_vec,jnp.zeros(2,dtype=jnp.float32),
    #     args=(vector_mat[:,0:2].reshape(N1,2),t1_final,p0_final,omega0+dw,sx),
    #     method='Nelder-Mead',options={'return_all':True}) 

    # log10_loss_list = [jnp.log10(loss_vec(x,vector_mat,t1_final,p0_final,omega0+dw,sx)) for x in res_vec.allvecs]
    # log10_loss_list_raw = [jnp.log10(loss_raw(x,t1_final,p0_final,omega0+dw,sx)) for x in res_raw.allvecs]
    # log10_loss_list_single = [[jnp.log10(loss_vec(x,vector_mat[:,i].reshape(N1,1),t1_final,p0_final,omega0+dw,sx)) 
    #         for x in res_single[i].allvecs]for i in range(3)]
    # log10_loss_list_double = [jnp.log10(loss_vec(x,vector_mat[:,0:2].reshape(N1,2)
    #     ,t1_final,p0_final,omega0+dw,sx)) for x in res_double.allvecs]

    # plt.plot(np.array(range(res_vec.nit)) , log10_loss_list, label='Three components')
    # plt.plot(np.array(range(res_raw.nit)), log10_loss_list_raw, label='All components')
    # plt.plot(np.array(range(res_double.nit)), log10_loss_list_double, label='Two components')

    # for i in range(3):
    #     print(res_single[i].success)
    #     plt.plot(np.array(range(res_single[i].nit)), log10_loss_list_single[i], label=f'Single component {i}')

    # plt.xlabel('iterations')
    # plt.ylabel('log10 loss')
    # plt.title('Infidelity evolution')

    # t_list = np.linspace(0,t1_final,100)
    # A_p= vmap(A,in_axes=(0,None,None),out_axes=0)
    # A_list = A_p(t_list,p0_final,t1_final)
    # A_dw = A_p(t_list, vector_mat@res_vec.x, t1_final)
    # print(res_vec.x)

    # plt.plot(t_list,A_list.real,label='Initial solution')
    # plt.plot(t_list,A_dw.real,label='$\delta A$')


    # plt.legend()
    # plt.show()

