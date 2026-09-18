import numpy as np
from scipy.sparse import diags,csc_matrix

####
def matrix(N,m,c,k):
    ####mass m, spring constant k, dampling constant c.
    n= 2*N

    J_upper = np.zeros(n-1)
    J_upper[::2] = 1
    J_lower = -J_upper
    J_main = np.zeros(n)
    J_diagonals = [J_lower, J_main, J_upper]
    J_positions = [-1, 0, 1]  # Positions of diagonals: -1 (lower), 0 (main), 1 (upper)
    J = csc_matrix(diags(J_diagonals, J_positions))


    R_diag = np.zeros(n)
    R_diag[:] = c
    R_diag[::2] = 0
    R = csc_matrix(diags(R_diag, offsets=0, shape=(n, n)))

    # Matrix B construction with sparse methods
    Q_diag = np.zeros(n)
    Q_diag[:] = 1/m
    Q_diag[::2] = 2*k
    Q_diag[0]=k

    Q_upper = np.zeros(n-2)
    Q_upper[:] = 0
    Q_upper[::2] = -k
    Q_lower = Q_upper
    Q_diagonals = [Q_lower, Q_diag, Q_upper]
    Q_positions = [-2, 0, 2]
    Q = csc_matrix(diags(Q_diagonals, Q_positions, shape=(n, n)))

    B_data = np.zeros((n,))  # Create a zero vector of size 2N
    B_data[1] = 1  # Set the N+1 (1-based index) entry, N in 0-based index, to 1
    B = csc_matrix(B_data).T

    return J,R,Q,B


