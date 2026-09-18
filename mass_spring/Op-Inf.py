import numpy as np
import cvxpy as cp
from utils import time_diff,time_diff_3,error_L2
from matrix import matrix
from ROM_solution import ROM,u_input
import matplotlib.pyplot as plt
from FOM_solution import FOM

dt = 0.001
m=4
k=4
c=1
N = 100
T = 10
Nt = int(T/dt)
####### consider smooth or nonsmooth input types
input_type = 'smooth'
######three types 'PH-OpInf_R', 'PH-OpInf_W' and 'sp'
type = 'sp'


J, R, Q,B = matrix(N, m,c,k)

X,F,Y,total_time = FOM(N,dt,T,input_type=input_type)

FOM_X = X
FOM_F = F
FOM_Y = Y

X_norm = error_L2(FOM_X,dt)
Y_norm = error_L2(FOM_Y,dt)

X_dot = time_diff_3(FOM_X,dt)

Vx,Sx,Ux = np.linalg.svd(FOM_X)

bb = np.sum(Sx**2)

for i in range(4):
    aa = np.sum(Sx[:5*(i+1)]**2)
    precent = aa/bb
    print(f'r={5*(i+1)} energy percent', precent)

def op_infer(r,type='sp'):
    #

    Vr = Vx[:,:r]

    Xr = Vr.T @ X_dot
    Fr = Vr.T @ FOM_F

    error_proj_X = error_L2(FOM_X - Vr@Vr.T@FOM_X,dt)
    error_proj_F = error_L2(FOM_F - Vr @ Vr.T @ FOM_F,dt)
    print("The projection error with r=%i" % r, error_proj_X,error_proj_F )


    # ##############If input is 0.1sin(x)

    xt = np.linspace(0, T, Nt + 1)
    u=u_input(xt,input_type = input_type)
    uu = u.reshape(1, -1)


    if type =='PH-OpInf_R':
        lamb =1e-11
        ########Least square optimization
        BB = np.linalg.solve(Fr @ Fr.T + lamb * np.identity(r), Fr @ FOM_Y.T)
        DD = cp.Variable((r,r))
        constrains = [(DD+DD.T)/2<<0]
        objective = cp.Minimize(cp.norm(Xr - DD @ Fr - BB @uu, 'fro') ** 2)
        prob = cp.Problem(objective, constrains)
        prob.solve(solver=cp.MOSEK, accept_unknown=True)
        Dr= DD.value
        Br = BB.flatten()


    if type == 'PH-OpInf_W':

        lamb = 1e5
        DD = cp.Variable((r, r))
        BB = cp.Variable((r,1))
        constrains = [(DD+DD.T)/2<<0]
        objective = cp.Minimize(cp.norm(Xr - DD @ Fr - BB@uu , 'fro')**2+ lamb*cp.norm(FOM_Y - BB.T@Fr,'fro')**2)
        prob = cp.Problem(objective,constrains)
        prob.solve(solver=cp.MOSEK,accept_unknown=True)
        Dr = DD.value
        BBB = BB.value

        Br = BBB.flatten()


    if type == 'sp':

        Jr = Vr.T @ J @ Vr
        Rr = Vr.T @ R @ Vr
        Dr = Jr-Rr
        Br = Vr.T @ B
        Br = Br.flatten()

    Jrsp = Vr.T @ J @ Vr
    Rrsp = Vr.T @ R @ Vr
    Drsp = Jrsp - Rrsp
    Brsp = Vr.T @ B
    Brsp = Brsp.flatten()


    Brr = Br.reshape(-1, 1)
    Brrsp = Brsp.reshape(-1, 1)

    error_opt_X = error_L2(Xr - Dr @ Fr - Brr @ uu,dt)
    error_opt_Y = error_L2(FOM_Y - Brr.T @ Fr,dt)


    print("The optimial error with r=%i" % r, error_opt_X, error_opt_Y)


    Xr_inf,Yr_inf,total_time = ROM(N, dt, T, r,Vr, Dr, Br)

    diff_X = Xr_inf - FOM_X[:, :]
    diff_Y = Yr_inf - FOM_Y[:, :]
    #

    error_X = error_L2(diff_X,dt)/error_L2(FOM_X,dt)
    error_Y = error_L2(diff_Y,dt)/error_L2(FOM_Y,dt)
    #
    print("The error with r=%i" % r,total_time, error_X, error_Y)
    #
    return error_proj_X,error_proj_F,error_opt_X, error_opt_Y,error_X,error_Y,total_time

error_X_all =[]
error_Y_all =[]
error_proj_all_X = []
error_proj_all_F=[]
error_opt_all_X = []
error_opt_all_Y = []
#
for i in range(10):

    error_proj_X,error_proj_F,error_opt_X,error_opt_Y,error_X,error_Y,total_time=op_infer(5*(i+1),type=type)

    error_X_all.append(error_X)
    error_Y_all.append(error_Y)
    error_proj_all_X.append(error_proj_X)
    error_proj_all_F.append(error_proj_F)
    error_opt_all_X.append(error_opt_X)
    error_opt_all_Y.append(error_opt_Y)

plt.semilogy(np.linspace(0, 50, len(error_opt_all_X)), error_opt_all_X, color='orange', linestyle='dashdot', marker='d',
         label='$ϵ_{optx}$')
plt.semilogy(np.linspace(0, 50, len(error_opt_all_Y)), error_opt_all_Y, color='red', linestyle='dashdot', marker='s',
         label='$ϵ_{opty}$')
plt.legend()
plt.xlabel('r')
plt.ylabel('Optimization Error')
plt.title('GP-OpInf')

plt.show()
#

plt.semilogy(np.linspace(0, 50, len(error_X_all)), error_X_all, color='blue', linestyle='dashdot', marker='o', label='$ϵ_X$')
plt.semilogy(np.linspace(0, 50, len(error_Y_all)), error_Y_all, color='black', linestyle='dashdot', marker='p', label='$ϵ_Y$')
plt.legend()
plt.xlabel('r')
plt.ylabel('Opt_Error')
plt.title('GP-OpInf')
plt.show()

plt.semilogy(np.linspace(0, 50, len(error_proj_all_X)), error_proj_all_X, color='blue', linestyle='dashdot', marker='o', label='$ϵ_{projx}$')
plt.semilogy(np.linspace(0, 50, len(error_proj_all_F)), error_proj_all_F, color='black', linestyle='dashdot', marker='p', label='$ϵ_{projF}$')
plt.legend()
plt.xlabel('r')
plt.ylabel('Projection Error')
plt.title('GP-OpInf')
plt.show()
#
#
#

