from mumps import DMumpsContext, spsolve
import numpy as np
from scipy import sparse
from scipy.linalg import norm
from GlobalSolvers import *  # Assuming this imports necessary matrices and functions
def Global_matrices(s, link="C:/Users/allo/Desktop/Vanguard/Matrices/"): # finite element
    A = loadmat(link+'A.mat')["A"]
    B = loadmat(link+'B.mat')["B"][2:]
    Q = loadmat(link+'Q.mat')["Q"][2:, 2:]
    m, n = B.shape
    X = np.ones((m + n, s))
    X0 = np.zeros((n+m, s))
    F = A@X[:n] + B.T@X[n:]
    G = B@X[:n]

    return A, B, F, G, Q, X0
A, B, F, G, Q, X0 = Global_matrices(s=200, link="/mnt/c/Users/allo/Desktop/Vanguard/Matrices/Navier Stokes/Channel Domain/Newton/L/L 8/")
I = csc_matrix(np.eye(Q.shape[0]))
Q_inv1 = sparse.diags(1/((B@B.T).diagonal()))
Q_inv2 = sparse.diags(1/(Q.diagonal()))
def Global_Mumps_Solver(A, F):
    F = csc_matrix(F)
    
    # Prepare data for MUMPS
    F_coo = F.tocoo()
    nz_rhs = F.count_nonzero()  # Number of non-zero entries in F
    nrhs = F.shape[1]  # Number of right-hand side vectors
    lrhs = F.shape[0]  # Leading dimension of the RHS
    rhs_sparse = F_coo.data.astype(np.float64)  # Ensure it's float64
    irhs_sparse = F_coo.row + 1  # Row indices (1-based for MUMPS)
    irhs_ptr = np.zeros(nrhs + 1, dtype=np.int32)  # Pointer array
    current_position = 1  
    rhs_col_indices = F_coo.col
    rhs = F.toarray().astype(np.float64)
    
    # Populate irhs_ptr for MUMPS
    for i in range(nrhs):
        col_nonzero_indices = np.where(rhs_col_indices == i)[0]
        irhs_ptr[i] = current_position
        current_position += len(col_nonzero_indices)
    
    irhs_ptr[nrhs] = nz_rhs + 1  # Ending pointer
    
    # Set up MUMPS context
    ctx = DMumpsContext()
    ctx.set_icntl(35, 2)  # Activate Block Low Rank Factorization
    ctx.set_icntl(27, 0)  # Control verbosity
    ctx.set_icntl(21, 0)  # Centralized solution
    ctx.set_icntl(20, 1)  # Sparse right-hand side
    ctx.set_icntl(7, 6)   # Approximate Minimum Degree with automatic quasi-dense row detection QAMD
    #ctx.set_icntl(48, 1)  # Improved multithreading using tree parallelism
    # Set up the matrix A
    ctx.set_centralized_sparse(A)
    
    # Set RHS information for MUMPS
    ctx.id.nrhs = nrhs  # Set number of RHS vectors
    ctx.id.nz_rhs = nz_rhs  # Set number of non-zero entries
    ctx.id.lrhs = lrhs  # Set leading dimension of the RHS
    ctx.id.irhs_ptr = ctx.cast_array(irhs_ptr)  # Cast pointer array
    ctx.id.irhs_sparse = ctx.cast_array(irhs_sparse)  # Cast row indices
    ctx.id.rhs_sparse = ctx.cast_array(rhs_sparse)  # Cast RHS sparse values
    ctx.id.rhs = ctx.cast_array(rhs)
    # Run the analysis, factorization and solve
    ctx.set_silent()
    ctx.run(job=6)
    
    # Retrieving the solution
    if ctx.myid == 0:
        x = rhs  # Get the solution from MUMPS
        
    else:
        x = None
    # Clean up
    ctx.destroy()  # Free memory
    return x

def Diagonal_preconditioner(A, B, Q, v_1, v_2):
    z_1 = Global_Mumps_Solver(A, v_1)
    z_2 = Global_Mumps_Solver(Q, v_2)
    return z_1, z_2

def Triangular_preconditioner(A, B, Q, v_1, v_2):
    z_2 = Global_Mumps_Solver(Q, v_2)
    z_1 = Global_Mumps_Solver(A, v_1-B.T@z_2)
    return z_1, z_2

def regularized_preconditioner(A, B, Q_inv, v_1, v_2, eps, alpha):

    z_1 = Global_Mumps_Solver(A - (eps/alpha)*(B.T@Q_inv@B), v_1 - B.T@Q_inv@v_2/alpha)

    z_2 = Q_inv@(v_2 - eps*B@z_1)/alpha

    return z_1, z_2

def Preconditioned_Global_Saddle_GMRES_restart(preconditioner, A, B, F, G, Q, X0, h=10, tol=1e-12, max_iter = 500, *args, **kwargs):
  start = time.time()
  s = F.shape[1]
  m, n = B.shape
  eps = kwargs.get('eps')
  if eps is None:
    eps = 1
  V = np.zeros(shape=(h+1, n+m, F.shape[1]))
  H = np.zeros(shape=(h+1, h))
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], (G - B@X0[:n])*eps))
  R0[:n], R0[n:] = preconditioner(A, B, Q, R0[:n], R0[n:], *args, **kwargs)
  beta0 = norm(R0, ord="fro")
  k = 0
  e_1 = np.zeros(h+1)
  e_1[0] = 1
  convergence = False
  beta = beta0
  Is = csc_matrix(np.eye(s))
  X_init = X0.copy()
  while not convergence:
      V[0] = R0/norm(R0, ord='fro')
      for j in range(h):
          W = np.zeros(shape=(n+m, s))
          W[:n],W[n:] = preconditioner(A, B, Q, A@V[j,:n] + B.T@V[j,n:], eps*B@V[j,:n], *args, **kwargs)
          for i in range(j+1):
              H[i, j] = np.trace(V[i].T@W)
              W -= H[i, j]*V[i]

          H[j+1, j] = norm(W, ord="fro")
          V[j+1] = W/H[j+1, j]
      V_h = V[:h].transpose(1, 0, 2).reshape(n+m, h*s)
      y = QR_solve(H, beta * e_1).reshape(-1, 1) # QR
      X = X_init + V_h@kron(y, Is)
      R = np.concatenate((F - A@X[:n] - B.T@X[n:], (G - B@X[:n])*eps))
      R[:n], R[n:] = preconditioner(A, B, Q, R[:n], R[n:],*args, **kwargs)
      beta = norm(R, ord="fro")
      if (beta/beta0 < tol) or (k >= max_iter):
          convergence = True
      else:
        X_init = X
        R0 = R
        k = k+1

  return X, k+1, beta, beta/beta0, time.time() - start

iterations = {}
runtimes = {}
relative_residuals = {}
residuals = {}

solvers = [#'GMRES',
           'DGMRES','TGMRES','RGMRES'
           #"I", "diag(BB.T)", "diag(S)"
           ]

for sol in solvers:
  iterations[sol] = []
  runtimes[sol] = []
  relative_residuals[sol] = []
  residuals[sol] = []

saddle_solvers = [#Global_Saddle_GMRES_restart,
                  lambda A, B, F, G, Q, X0 : Preconditioned_Global_Saddle_GMRES_restart(Diagonal_preconditioner, A, B, F, G, Q, X0),
                  lambda A, B, F, G, Q, X0 : Preconditioned_Global_Saddle_GMRES_restart(Triangular_preconditioner, A, B, F, G, Q, X0),
                  lambda A, B, F, G, Q, X0, *args, **kwargs : Preconditioned_Global_Saddle_GMRES_restart(regularized_preconditioner, A, B, F, G, Q, X0, *args, **kwargs),
                  #lambda A, B, F, G, Q, X0, *args, **kwargs : Preconditioned_Global_Saddle_GMRES_restart(regularized_preconditioner, A, B, F, G, Q_inv1, X0, *args, **kwargs),
                  #lambda A, B, F, G, Q, X0, *args, **kwargs : Preconditioned_Global_Saddle_GMRES_restart(regularized_preconditioner, A, B, F, G, Q_inv2, X0, *args, **kwargs)
                  ]
L = [5, 6, 7, 8]
Grid = [3, 4, 5, 6]
Alpha = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


for grid in Grid:
  print(grid)
  A, B, F, G, Q, X0 = Global_matrices(s=100, link = f"/mnt/c/Users/allo/Desktop/Vanguard/Matrices/Navier Stokes/Channel Domain/Newton/grid/grid {grid}/")
  Q_inv = sparse.diags(1/((B@B.T).diagonal()))
  for i, saddle in enumerate(saddle_solvers):
    if solvers[i] == "RGMRES":
        x, k, Resid, Relative_resid, runtime = saddle(A, B, F, G, Q_inv, X0, eps = -1, alpha=1e-6)
    else:
        x, k, Resid, Relative_resid, runtime = saddle(A, B, F, G, Q, X0)
    sol = solvers[i]
    iterations[sol].append(k)
    runtimes[sol].append(runtime)
    relative_residuals[sol].append(Relative_resid)
    residuals[sol].append(Resid)

import pandas as pd
parameter = {"Grid" : Grid}
df_iter = pd.DataFrame({**parameter, **iterations})
df_runtime = pd.DataFrame({**parameter, **runtimes}).map(lambda x: round(x, 2))
df_relative_residuals = pd.DataFrame({**parameter, **relative_residuals}).map(lambda x: '{:.1e}'.format(x))
df_residuals = pd.DataFrame({**parameter, **residuals}).map(lambda x: float(x)).map(lambda x: '{:.1e}'.format(x))


with pd.ExcelWriter('Global Grid Newton Full.xlsx') as writer:
    # Write each DataFrame to a separate sheet
    df_iter.to_excel(writer, sheet_name='Iterations', index=False)
    df_runtime.to_excel(writer, sheet_name='Run time', index=False)
    df_relative_residuals.to_excel(writer, sheet_name='Relative Residuals', index=False)
    df_residuals.to_excel(writer, sheet_name='Residuals', index=False)
