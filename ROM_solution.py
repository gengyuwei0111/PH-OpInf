import numpy as np
from matrix import matrix
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
def ROM(N,dt,T,r,phi,Dr,Br,input_type="smooth"):

    N_t = int(T/dt)

    J, R, Q, B = matrix(N, m, c, k)
    # D = J - R
    Qr = phi.T@Q@phi

    X_0=np.zeros(2*N)

    Y = np.zeros((1,N_t+1))
    Y[:,0] = 0

    X = np.zeros((2*N,N_t+1))
    X[:,0] = X_0

    F = np.zeros((2*N,N_t+1))
    F [:,0] = Q@X_0

    Xr0 = phi.T@X_0

    D1 = (np.eye(r)) - 0.5 * dt * Dr @ Qr
    D2 = (np.eye(r)) + 0.5 * dt * Dr @ Qr
    B1 = np.linalg.solve(D1,D2)
    B3 = np.linalg.solve(D1,dt*Br)

    C1  =  Br.T@Qr
    start = time.time()

    for i in range(N_t):
        Xr_next = B1@Xr0 + B3 * u_input((2*i+1)*dt/2, input_type)
        Xr0  = Xr_next
        X[:, i + 1] = phi @Xr_next
        Y[:, i + 1] = C1@Xr_next
    stop = time.time()

    total_time = stop - start
    return X,Y,total_time

