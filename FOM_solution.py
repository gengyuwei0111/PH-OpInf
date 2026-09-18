import numpy as np
from matrix import matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time
################################


def u_input(t, input_type="smooth"):
    if input_type == "smooth":
        return np.exp(-t / 2) * np.sin(t**2)

    elif input_type == "nonsmooth":
        max_k = int(np.max(t) // 2)

        conditions = [
            (t >= 2 * k) & (t < 2 * (k + 1))
            for k in range(max_k + 1)
        ]

        functions = [
            lambda t, k=k: t - (2 * k + 1)
            for k in range(max_k + 1)
        ]

        return np.piecewise(t, conditions, functions)

    else:
        raise ValueError("input_type must be 'smooth' or 'nonsmooth'")

#################################
m=4
k=4
c=1
#################################


def FOM(N,dt,T,input_type='smooth'):

    N_t = int(T/dt)

    J, R, Q,B = matrix(N,m,c,k)

    D = J-R

    X_0 = np.zeros(2 * N)

    Y = np.zeros((1, N_t + 1))
    Y[:, 0] = 0

    X = np.zeros((2 * N, N_t + 1))
    X[:, 0] = X_0

    F = np.zeros(((2 * N, N_t + 1)))
    F[:, 0] = Q @ X_0



    D1 = csc_matrix(np.eye(2*N)) - 0.5*dt*D@Q
    D2 = csc_matrix(np.eye(2*N)) + 0.5*dt*D@Q
    B1 = spsolve(D1,D2)
    B3 = spsolve(D1, dt *B)
    start = time.time()

    X_n = X_0
    for i in range(N_t):
        X_next = B1@X_n + B3*u_input((2*i+1)*dt/2,input_type=input_type)
        F_next = Q@X_next

        X_n = X_next
        X[:, i + 1] = X_next
        F[:, i + 1] = F_next
        Y[:,i+1] = B.T@F[:, i + 1]

    stop = time.time()
    total_time = stop - start
    return X,F,Y,total_time

# #
# N = 100
# T=10
# dt = 0.001
# input_type = "smooth"
# J, R, Q,B = matrix(N,m,c,k)
# X,F,Y,total_time = FOM(N,dt,T,input_type=input_type)
#
# print(total_time)
