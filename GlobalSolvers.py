# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:04:22 2024

@author: allo
"""

# Importing Libraries
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from scipy.sparse import csc_matrix, diags, kron, block_diag, vstack
import time
from scipy.sparse.linalg import spilu
from StokesSolvers import QR_solve
np.random.seed(155)


    

def Global_matrices(s, link="C:/Users/allo/Desktop/Vanguard/Matrices/"): # finite element

    A = loadmat(link+'A.mat')["Ast"]
    B = loadmat(link+'B.mat')["Bst"][2:]
    Q = loadmat(link+'Q.mat')["Q"][2:, 2:]
    F = A@np.ones((A.shape[0], s)) + B.T@np.ones((B.shape[0], s)) 
    G = B@np.ones((A.shape[0], s)) 
    m, n = B.shape
    X0 = np.zeros((n+m, s))

    return A, B, F, G, Q, X0

def Global_saddle_matrices(n, m, s): # finite difference
    h = 1/(m + 1)
    I = csc_matrix(np.identity(n))
    main_diag = np.ones(m)
    lower_diag = -1 * np.ones(m - 1)
    upper_diag = np.zeros(m - 1)
    
    tridiag_matrix = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')
    Psi = tridiag_matrix / h
    
    main_diag = 2 * np.ones(m)  
    lower_diag = -1 * np.ones(m - 1)  
    upper_diag = -1 * np.ones(m - 1) 
    
    tridiag_matrix = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')
    Phi = tridiag_matrix / (h**2)
        
    A = block_diag((kron(Phi, I) + kron(I, Phi), kron(Phi, I) + kron(I, Phi)))
    
    B = vstack([kron(I, Psi), kron(Psi, I)]).T
    
    Q = B@B.T
    
    m, n = B.shape
    
    F = np.zeros(shape=(n,s))+1
    G = np.zeros(shape=(m,s))
    X0 = np.ones((n+m, s))
    
    return A, B, F, G, Q, X0

    

def Arnoldi(A, B, f, g, x0, k):
    m, n = B.shape
    r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
    v = np.zeros((k+1, n+m))
    v[0] = r0 / norm(r0)
    H = np.zeros((k+1,k))
    for j in range(k):
        w = np.concatenate((A@v[j,:n] + B.T@v[j, n:], B@v[j, :n]))
        for i in range(j+1):
            H[i,j] = v[i].T@w
            w -= H[i,j]*v[i]


        H[j+1,j] = norm(w)
        v[j+1] = w/H[j+1,j]
    
    return v, H




def Global_Arnoldi(A, B, F, G, X0, k):
    m, n = B.shape
    V = np.zeros(shape=(k+1, n+m, F.shape[1]))
    H = np.zeros(shape=(k+1, k))
    R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
    V[0] = R0/norm(R0, ord='fro')
    for j in range(k):
        W = np.concatenate((A@V[j,:n] + B.T@V[j,n:], B@V[j,:n]))
        for i in range(j+1):
            H[i, j] = np.trace(V[i].T@W)
            W -= H[i, j]*V[i]
        
        H[j+1, j] = norm(W, ord="fro")
        V[j+1] = W/H[j+1, j]     
    return V, H





def Global_Saddle_GMRES_restart(A, B, F, G, Q, X0, h=10, tol=1e-6,max_iter = 2000):
    
  start = time.time()
  s = F.shape[1]
  m, n = B.shape
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
  beta0 = norm(R0, ord="fro")
  k = 0
  e_1 = np.zeros(h+1)
  e_1[0] = 1
  convergence = False
  beta = beta0
  Is = csc_matrix(np.eye(s))
  X_init = X0.copy()
  while not convergence:
      V, H = Global_Arnoldi(A, B, F, G, X_init, h)
      V_h = V[:h].transpose(1, 0, 2).reshape(n+m, h*s)
      y = QR_solve(H, beta * e_1).reshape(-1, 1) # QR
      X = X_init + V_h@kron(y, Is)
      R = np.concatenate((F - A@X[:n] - B.T@X[n:], G - B@X[:n]))
      beta = norm(R, ord="fro")
      if (beta/beta0 < tol) or (k >= max_iter):
          convergence = True
      else:
        X_init = X
        R0 = R
        k = k+1

  return X, k, beta, beta/beta0, time.time() - start






def Global_Saddle_GMRES(A, B, F, G, Q, X0, max_iter = 2000):

  s = F.shape[1]
  m, n = B.shape
  V = np.zeros(shape=(max_iter+1, n+m, s))
  H = np.zeros((max_iter+1,max_iter))
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
  beta0 = norm(R0, ord="fro")
  V[0] = R0 / beta0
  Is = csc_matrix(np.identity(s))
  X = X0.copy()
  for j in range(max_iter):
      
      W = np.concatenate((A@V[j,:n] + B.T@V[j,n:], B@V[j,:n]))
      
      for i in range(j+1):
          H[i, j] = np.trace(V[i].T@W)
          W -= H[i, j]*V[i]
      H[j+1, j] = norm(W, ord="fro")
      V[j+1] = W/H[j+1, j]     

     
  e_1 = np.zeros(max_iter+1)
  e_1[0] = 1
  V_h = V[:max_iter].transpose(1, 0, 2).reshape(n+m, max_iter*s)
  y = QR_solve(H, beta0 * e_1).reshape(-1, 1) # QR
  X = X0 + V_h@kron(y, Is)
  R = np.concatenate((F - A@X[:n] - B.T@X[n:], G - B@X[:n]))
  beta = norm(R, ord="fro")

  return X, beta/beta0


def preconditioned_Global_CG(A, F, L, max_iter=1000, tol = 1e-6):
    n, s = F.shape
    R = np.zeros((max_iter+1, n, s))
    Z = np.zeros((max_iter+1, n, s))
    P = np.zeros((max_iter+1, n, s))
    X = np.random.rand(n, s)
    alpha = np.zeros(max_iter+1)
    beta = np.zeros(max_iter+1)
    R[0] = F - A@X
    beta0 = norm(R[0], ord='fro')
    Z[0] = L.solve(R[0]) 
    P[0] = Z[0]
    k = 0
    convergence = False
    while not convergence:
        alpha[k] = np.trace(Z[k].T@R[k]) / np.trace((A@P[k]).T@P[k])
        X += alpha[k]*P[k]
        R[k+1] = R[k] - alpha[k]*(A@P[k])
        Z[k+1] = L.solve(R[k+1])
        beta[k] = np.trace(Z[k+1].T@R[k+1]) / np.trace(Z[k].T@R[k])
        P[k+1] = Z[k+1] + beta[k]*P[k]
        betak = norm(R[k], ord='fro')
        if (betak/beta0 < tol) or (k >= max_iter-1):
            convergence = True
        else:
            k += 1
        
    return X

def Diagonal_preconditioner(A, B, Q, La ,Lq ,v_1, v_2):
    z_1 = preconditioned_Global_CG(A, v_1, La)
    z_2 = preconditioned_Global_CG(Q, v_2, Lq)
    return z_1, z_2

def Triangular_preconditioner(A, B, Q, La ,Lq ,v_1, v_2):
    z_2 = preconditioned_Global_CG(Q, v_2, Lq)
    z_1 = preconditioned_Global_CG(A, v_1-B.T@z_2, La)
    return z_1, z_2

def regularized_preconditioner(A, B, Q, La ,Lq ,v_1, v_2, alpha=10):

    
    # P_1@y = v

    y_1 = v_1
    y = preconditioned_Global_CG(A, v_1, La)
    y_2 = v_2 - B@y

    # P_2@x = y
    
    x_1 = preconditioned_Global_CG(A, y_1, La)
    x_2 = preconditioned_Global_CG(Q, y_2/(alpha-1), Lq)
    
    # P_3@z = x

    z_1 = preconditioned_Global_CG(A, A@x_1 - B.T@x_2, La)
    z_2 = x_2


    return z_1, z_2


def Preconditioned_Global_Saddle_GMRES_restart(preconditioner, A, B, F, G, Q, X0, h=20, tol=1e-6,max_iter = 1000, *args, **kwargs):
  start = time.time()
  s = F.shape[1]
  m, n = B.shape
  La = spilu(A, drop_tol=1e-2)
  Lq = spilu(Q, drop_tol=1e-2)
  V = np.zeros(shape=(h+1, n+m, F.shape[1]))
  H = np.zeros(shape=(h+1, h))
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
  R0[:n], R0[n:] = preconditioner(A, B, Q, La, Lq, R0[:n], R0[n:], *args, **kwargs)
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
          W[:n],W[n:] = preconditioner(A, B, Q, La, Lq, A@V[j,:n] + B.T@V[j,n:], B@V[j,:n], *args, **kwargs)
          for i in range(j+1):
              H[i, j] = np.trace(V[i].T@W)
              W -= H[i, j]*V[i]
          
          H[j+1, j] = norm(W, ord="fro")
          V[j+1] = W/H[j+1, j]  
      V_h = V[:h].transpose(1, 0, 2).reshape(n+m, h*s)
      y = QR_solve(H, beta * e_1).reshape(-1, 1) # QR
      X = X_init + V_h@kron(y, Is)
      R = np.concatenate((F - A@X[:n] - B.T@X[n:], G - B@X[:n]))
      R[:n], R[n:] = preconditioner(A, B, Q, La, Lq, R[:n], R[n:], *args, **kwargs)
      beta = norm(R, ord="fro")
      print(beta/beta0)
      if (beta/beta0 < tol) or (k >= max_iter):
          convergence = True
      else:
        X_init = X
        R0 = R
        k = k+1

  return X, k, beta, beta/beta0, time.time() - start


def FGMRES(preconditioner, A, B, F, G, Q, X0, max_iter = 1000, *args, **kwargs):

  s = F.shape[1]
  m, n = B.shape
  La = spilu(A, drop_tol=1e-2)
  Lq = spilu(Q, drop_tol=1e-2)
  V = np.zeros(shape=(max_iter+1, n+m, s))
  H = np.zeros((max_iter+1,max_iter))
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
  beta0 = norm(R0, ord="fro")
  V[0] = R0 / beta0
  Is = csc_matrix(np.identity(s))
  X = X0.copy()
  for j in range(max_iter):
      
      Z = np.zeros(shape=(n+m, s))
      Z[:n], Z[n:] = preconditioner(A, B, Q, La, Lq, V[j,:n], V[j,n:], *args, **kwargs)
      W = np.concatenate((A@Z[:n] + B.T@Z[n:], B@Z[:n]))
      
      for i in range(j+1):
          H[i, j] = np.trace(V[i].T@W)
          W -= H[i, j]*V[i]
      H[j+1, j] = norm(W, ord="fro")
      V[j+1] = W/H[j+1, j]     

     
  e_1 = np.zeros(max_iter+1)
  e_1[0] = 1
  V_h = V[:max_iter].transpose(1, 0, 2).reshape(n+m, max_iter*s)
  y = QR_solve(H, beta0 * e_1).reshape(-1, 1) # QR
  X = X0 + V_h@kron(y, Is)
  R = np.concatenate((F - A@X[:n] - B.T@X[n:], G - B@X[:n]))
  beta = norm(R, ord="fro")
  
  return X, beta/beta0

def Preconditioned_Global_FGMRES(preconditioner, A, B, F, G, Q, X0, h=20, tol=1e-6,max_iter = 1000, *args, **kwargs):
  start = time.time()
  s = F.shape[1]
  m, n = B.shape
  La = spilu(A, drop_tol=1e-2)
  Lq = spilu(Q, drop_tol=1e-2)
  V = np.zeros(shape=(h+1, n+m, s))
  H = np.zeros(shape=(h+1, h))
  R0 = np.concatenate((F - A@X0[:n] - B.T@X0[n:], G - B@X0[:n]))
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
          Z = np.zeros(shape=(n+m, s))
          Z[:n], Z[n:] = preconditioner(A, B, Q, La, Lq, V[j,:n], V[j,n:], *args, **kwargs)
          W = np.concatenate((A@Z[:n] + B.T@Z[n:], B@Z[:n]))
          for i in range(j+1):
              H[i, j] = np.trace(V[i].T@W)
              W -= H[i, j]*V[i]
          
          H[j+1, j] = norm(W, ord="fro")
          V[j+1] = W/H[j+1, j]  
      V_h = V[:h].transpose(1, 0, 2).reshape(n+m, h*s)
      y = QR_solve(H, beta * e_1).reshape(-1, 1) 
      X = X_init + V_h@kron(y, Is)
      R = np.concatenate((F - A@X[:n] - B.T@X[n:], G - B@X[:n]))
      beta = norm(R, ord="fro")
      if (beta/beta0 < tol) or (k >= max_iter):
          convergence = True
      else:
        X_init = X
        R0 = R
        k = k+1

  return X, k, beta, beta/beta0, time.time() - start

"""A, B, F, G, Q, X0 = Global_matrices(s=10)
X, k, res, relative_res, runtime = Preconditioned_Global_Saddle_GMRES_restart(Triangular_prefunction, A, B, F, G, Q, X0, h=20, tol=1e-9,max_iter = 1000)
print(f'Number of Iterations :{k}')"""

# Saddle MINRES

def Global_MINRES(A, B, F, G, Q, x0, max_iter=2000, tol=1e-6):
    start = time.time()
    m, n = B.shape
    s = F.shape[1]
    x = x0.copy()
    v = np.zeros((max_iter+1, n+m, s))
    w = np.zeros((max_iter+1, n+m, s))
    s = np.zeros(max_iter+1)
    c = np.zeros(max_iter+1)
    gamma = np.zeros(max_iter+1)
    delta = np.zeros(max_iter+1)
    v[1,:n] = F - A@x[:n] - B.T@x[n:]
    v[1,n:] = G - B@x[:n]
    gamma[1] = norm(v[1], ord="fro")
    beta0 = gamma[1].copy()
    eta = gamma[1]
    s[0] = s[1] = 0
    c[0] = c[1] = 1
    j = 1
    convergence = False
    while not convergence:
        v[j] /= gamma[j]
        delta[j] = np.trace((np.concatenate((A@v[j, :n]+B.T@v[j, n:],B@v[j, :n]))).T@v[j])
        v[j+1] = np.concatenate((A@v[j, :n]+B.T@v[j, n:] , B@v[j, :n])) - delta[j]*v[j] - gamma[j]*v[j-1]
        gamma[j+1] = norm(v[j+1], ord='fro')
        alpha_0 = c[j]*delta[j] - c[j-1]*s[j]*gamma[j]
        alpha_1 = np.sqrt(alpha_0**2 + gamma[j+1]**2)
        alpha_2 = s[j]*delta[j] + c[j-1]*c[j]*gamma[j]
        alpha_3 = s[j-1]*gamma[j]
        c[j+1] = alpha_0/alpha_1
        s[j+1] = gamma[j+1]/alpha_1
        w[j+1] = (v[j] - alpha_3*w[j-1] - alpha_2*w[j])/alpha_1
        x += c[j+1]*eta*w[j+1]
        eta = -s[j+1]*eta
        R = np.concatenate((F - A@x[:n] - B.T@x[n:], G - B@x[:n]))
        beta = norm(R, ord="fro")
        if (beta/beta0 < tol) or (j >= max_iter-1):
            convergence = True
        else:
          j += 1

    return x, j+1, beta, beta/beta0, time.time() - start




# Preconditioned MINRES


def preconditioned_MINRES(preconditioner, A, B, F, G, Q, x0, max_iter=1000, tol=1e-6, *args, **kwargs): # page 207
    start = time.time()
    m, n = B.shape
    La = spilu(A, drop_tol=1e-2)
    Lq = spilu(Q, drop_tol=1e-2)
    s = F.shape[1]
    x = x0.copy()
    v = np.zeros((max_iter+1, n+m, s))
    z = np.zeros((max_iter+1, n+m, s))
    w = np.zeros((max_iter+1, n+m, s))
    s = np.zeros(max_iter+1)
    c = np.zeros(max_iter+1)
    gamma = np.zeros(max_iter+1)
    gamma[0] = 1 # /0 ?
    delta = np.zeros(max_iter+1)
    v[1,:n] = F - A@x[:n] - B.T@x[n:]
    v[1,n:] = G - B@x[:n]
    beta0 = norm(v[1], ord='fro')
    z[1, :n], z[1, n:] = preconditioner(A, B, Q, La, Lq, v[1, :n], v[1, n:], *args, **kwargs) # solve Mz = v
    gamma[1] = np.sqrt(np.trace(v[1].T@z[1]))
    eta = gamma[1]
    s[0] = s[1] = 0
    c[0] = c[1] = 1
    j = 1
    convergence = False
    while not convergence:
        z[j] /= gamma[j]
        delta[j] = np.trace((np.concatenate((A@z[j, :n] + B.T@z[j, n:], B@z[j, :n]))).T@z[j])
        v[j+1] = np.concatenate((A@z[j, :n] + B.T@z[j, n:], B@z[j, :n])) - (delta[j]/gamma[j])*v[j] - (gamma[j]/gamma[j-1])*v[j-1]
        z[j+1, :n], z[j+1, n:] = preconditioner(A, B, Q, La, Lq, v[j+1, :n], v[j+1, n:], *args, **kwargs) # solve Mz = v
        gamma[j+1] = np.sqrt(np.trace(z[j+1].T@v[j+1]))
        alpha_0 = c[j]*delta[j] - c[j-1]*s[j]*gamma[j]
        alpha_1 = np.sqrt(alpha_0**2 + gamma[j+1]**2)
        alpha_2 = s[j]*delta[j] + c[j-1]*c[j]*gamma[j]
        alpha_3 = s[j-1]*gamma[j]
        c[j+1] = alpha_0/alpha_1
        s[j+1] = gamma[j+1]/alpha_1
        w[j+1] = (z[j] - alpha_3*w[j-1] - alpha_2*w[j])/alpha_1
        x += c[j+1]*eta*w[j+1]
        eta = -s[j+1]*eta
        R = np.concatenate((F - A@x[:n] - B.T@x[n:], G - B@x[:n]))
        beta = norm(R, ord="fro")
        print(beta/beta0)
        if (beta/beta0 < tol) or (j >= max_iter-1): 
            convergence = True
        else:
          j += 1

    return x, j+1, beta, beta/beta0, time.time() - start




# Splitting Methods

# Shift-splitting method

def SS(A, B, F, G, Q, x0, alpha=0.1, max_iter=1000, tol=1e-6): 
    m, n = B.shape
    convergence = False
    x = x0.copy()
    r1 = F - A@x[:n] - B.T@x[n:]
    r2 = G - B@x[:n]
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    I = csc_matrix(np.identity(n))
    A_alpha = alpha*I - A 
    B_alpha = alpha*I + A + B.T@B/alpha
    L = spilu(B_alpha, drop_tol=1e-2)
    while not convergence:
        rhs1 = F + (A_alpha@x[:n] - B.T@x[n:])/2
        rhs2 = -G + (B@x[:n] + alpha*x[n:])/2
        
        
        # step 1

        t1 = rhs1 - B.T@rhs2/alpha

        # step 2
        
        
        x[:n] = preconditioned_Global_CG(B_alpha, 2*t1, L)
        
        
        # step 3

        x[n:] = (B@x[:n] + 2*rhs2)/alpha
        
        
        r1 = F - A@x[:n] - B.T@x[n:]
        r2 = G - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0
    

# Hermitian and skew-Hermitian splitting method

def HSS(A, B, f, g, Q, x0, alpha, max_iter=1000, tol=1e-6): 
    E = B.T
    m, n = B.shape
    I = csc_matrix(np.identity(n))
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n]
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    A_alpha = alpha*I + A
    A_alpha_ = alpha*I - A
    E_alpha = alpha*I + E@E.T/alpha
    La = spilu(A_alpha, drop_tol=1e-2)
    Le = spilu(E_alpha, drop_tol=1e-2)
    while not convergence:
        
        # solve first system
        
        rhs1 = alpha * x[:n] - E@x[n:] + f
        rhs2 = E.T@x[:n] + alpha*x[n:] - g
        z1 = preconditioned_Global_CG(A_alpha, rhs1, La) 
        z2 = rhs2/alpha
        
        # solve second system

        rhs1 = f + A_alpha_@z1
        rhs2 = -g + alpha*z2
        
       # step1 
       
        t1 = rhs1 - E@rhs2/alpha
       
       # step2
        
        
        x[:n] = preconditioned_Global_CG(E_alpha, t1, Le)
       
       # step3
       
        x[n:] = (E.T@x[:n] + rhs2)/alpha
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        
    
    return x, k, beta/beta0


def HSS_preconditioner(A, B, Q, r_1, r_2, alpha): #alpha = 0.5, iter = 593
    
    n = len(r_1)
    I = csc_matrix(np.identity(n))
    A_alpha = alpha*I + A
    B_alpha = alpha*I + B.T@B/alpha
    La = spilu(A_alpha, drop_tol=1e-2)
    Lb = spilu(B_alpha, drop_tol=1e-2)
    v_1 = preconditioned_Global_CG(A_alpha, r_1, La)
    v_2 = r_2/alpha
    t = v_1 - (1/alpha)*B.T@v_2
    z_1 = preconditioned_Global_CG(B_alpha, t, Lb)
    z_2 = (B@z_1 + v_2)/alpha
    
    return z_1, z_2


