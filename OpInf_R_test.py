import numpy as np
import cvxpy as cp
from utils import error_L2,time_diff
from matrix import matrix
from ROM_solution import ROM, u_input
from FOM_solution import FOM
import matplotlib.pyplot as plt


dt = 0.01
m=4
k=4
c=1
N = 100
T = 10
Nt = int(T/dt)


J, R, Q,B = matrix(N, m,c,k)

####### consider smooth or nonsmooth input types
input_type = 'smooth'

X,F,Y,total_time = FOM(N,dt,T,input_type=input_type)

FOM_X = X
FOM_F = F
FOM_Y = Y


X_dot = time_diff(FOM_X, dt)

Vx, Sx, Ux = np.linalg.svd(FOM_X)


# lamb = [0,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6]
# lamb = np.array([0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
lamb = np.array([1e-14, 1e-13,1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7])


def op_infer(r):
    #
    phi = Vx[:, :r]

    Xr = phi.T @ X_dot
    Fr = phi.T @ FOM_F

    xt = np.linspace(0, T, Nt + 1)
    uu = u_input(xt, input_type = input_type)

    u = uu.reshape(1, -1)

    Error_opt_all_X = []
    Error_opt_all_Y = []
    Error_all_X = []
    Error_all_Y = []
    mag_B=[]
    mag_D=[]
    mag_opt_B=[]
    BB_norm_all= []


    for i in range(len(lamb)):
    ########Least square optimization
        BB = np.linalg.solve(Fr @ Fr.T + lamb[i] * np.identity(r), Fr @ FOM_Y.T)
        BB_norm = np.linalg.norm(BB,ord='fro')
        BB_norm_all.append(BB_norm)

        DD = cp.Variable((r, r))
        constrains = [(DD + DD.T) / 2 << 0]
        objective = cp.Minimize(cp.norm(Xr - DD @ Fr - BB @ u, 'fro') ** 2)
        prob = cp.Problem(objective, constrains)
        prob.solve(solver=cp.MOSEK, accept_unknown=True)

        Dr = DD.value
        Br = BB.flatten()

        Brr = Br.reshape(-1, 1)

        if Dr is None or np.max(np.abs(Dr))>=1e30:

            error_opt_X = error_L2(Xr - Dr @ Fr - Brr @ u,dt)
            error_opt_Y = error_L2(FOM_Y - Brr.T @ Fr,dt)

            Error_opt_all_X.append(error_opt_X)
            Error_opt_all_Y.append(error_opt_Y)
            Error_all_X.append(np.nan)
            Error_all_Y.append(np.nan)
            print("The error with r=%i" % r, np.nan, error_opt_Y)
            mag_D.append(np.max(np.abs(Dr)))
            mag_B.append(np.max(np.abs(Br)))
            mag_opt_B.append((np.max(np.abs(FOM_Y.T - Fr.T@Brr))))

        else:
            mag_D.append(np.max(np.abs(Dr)))
            mag_B.append(np.max(np.abs(Br)))
            mag_opt_B.append((np.max(np.abs(FOM_Y.T - Fr.T @ Brr))))

            Rr = -(1 / 2) * (Dr + Dr.T)
            eigenvalues = np.linalg.eigvals(Rr)
            print(f'eigenvalue with r={r}', np.min(eigenvalues))

            error_opt_X = error_L2(Xr - Dr @ Fr - Brr @ u,dt)
            error_opt_Y = error_L2(FOM_Y - Brr.T @ Fr,dt)

            print("The optimial error with r=%i" % r, error_opt_X, error_opt_Y)

            Error_opt_all_X.append(error_opt_X)
            Error_opt_all_Y.append(error_opt_Y)

            Xr_inf,Yr_inf,time_all = ROM(N, dt, T, r,phi, Dr, Br, input_type = input_type)

            diff_X = Xr_inf - FOM_X[:, :]
            diff_Y = Yr_inf - FOM_Y[:, :]

            error_X = error_L2(diff_X,dt)
            Error_all_X.append(error_X)

            error_Y = error_L2(diff_Y,dt)
            Error_all_Y.append(error_Y)

            print("The error with r=%i" % r, error_X, error_Y)


    return Error_opt_all_X, Error_opt_all_Y,Error_all_X,Error_all_Y,BB_norm_all



r=30

Error_opt_all_X, Error_opt_all_Y,Error_all_X,Error_all_Y,BB_norm_all=op_infer(r)


plt.figure(figsize=(7,5))
plt.loglog(lamb, Error_opt_all_X, marker='o', label=r'$\mathcal{E}_{opt,X}$')
plt.loglog(lamb, Error_opt_all_Y, marker='s', label=r'$\mathcal{E}_{opt,Y}$')
plt.loglog(lamb, Error_all_X, marker='^', label=r'$\mathcal{E}_{X}$')
plt.loglog(lamb, Error_all_Y, marker='d', label=r'$\mathcal{E}_{Y}$')
plt.xlabel(r'$\lambda$')
plt.ylabel('Error')
plt.title(f'Errors vs regularization parameter with r{r}')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
