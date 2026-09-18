import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
def time_diff(u,dt):
    n=len(u[0])
    N = len(u)
    U_FD = np.zeros((N,n))
    for i in range(1, n - 1):
        U_FD[:, i] =0.5 * (u[:, i + 1] - u[:, i - 1])/dt
    U_FD[:, 0] = 0.5 * (4 *u[:, 1] - 3 * u[:, 0] - u[:, 2])/dt
    U_FD[:, -1] = 0.5 * (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3])/dt

    return U_FD

def time_diff_3(u,dt):
    Nt=len(u[0])
    N = len(u)

    U_FD = np.zeros((N,Nt))
    for i in range(2, Nt - 2):
        U_FD[:, i] =(-u[:,i+2]+8*u[:, i + 1] -8* u[:, i - 1]+u[:,i-2])/dt/12
    U_FD[:, 0] = (u[:, 1] - u[:, 0])/dt
    U_FD[:,1] = (u[:, 2] - u[:, 0])/dt/2
    U_FD[:, -1] = (-u[:, -2]+u[:, -1])/dt
    U_FD[:,-2] =  (-u[:, -3]+u[:, -1])/dt/2


    return U_FD

def error_L2(x,dt):


    # if x.ndim==1:
    #     err = np.sqrt(np.sum(np.abs(x)**2) *dt)
    # elif x.ndim ==2:
    #     if x.shape[0] == 1:
    #         err = np.sqrt(np.sum(np.abs(x)**2)*dt)
    #     else:
    #         err_L2=np.sum(x**2,0)
    #         err = np.sqrt(np.sum(err_L2)*dt)
    err_L2 = np.sum(x ** 2, 0)
    err = np.sqrt(np.sum(err_L2) * dt)
    return err


def update_SVD(U,S,V,B,r):


    Res= B - U@U.T@B

    Q,R = np.linalg.qr(Res,mode='reduced')
    W=np.vstack((np.hstack((np.diag(S),U.T@B)),np.hstack((np.zeros((len(R),len(S))),R))))

    UU,S_new,VV  = svds(W,r)

    U_new = np.hstack((U,Q))@UU
    V_new = block_diag(V,np.eye(len(VV[0])-len(V)))

    return U_new, S_new , V_new

