# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:42:56 2024

@author: allo
"""

# Importing Libraries
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm, qr
import scipy
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix, csr_matrix, diags, kron, block_diag, vstack
from scipy.sparse.linalg import cg, spsolve_triangular, spilu
import scipy.sparse as ssp

#import ilupp
import time

np.random.seed(155)

def condition_number(A):
    lambda_min = scipy.sparse.linalg.eigs(A , k=1, which='SM', return_eigenvectors=False)[0]
    lambda_max = scipy.sparse.linalg.eigs(A , k=1, which='LM', return_eigenvectors=False)[0]
    return norm(lambda_max)/norm(lambda_min)

# finite difference matrices

def saddle_matrices(n, m):
    h = 1/(m + 1)
    I = csc_matrix(np.identity(n))
    main_diag = np.ones(m)
    lower_diag = -1 * np.ones(m - 1)
    upper_diag = np.zeros(m - 1)
    
    tridiag_matrix = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csr')
    Psi = tridiag_matrix / h
    
    main_diag = 2 * np.ones(m)  
    lower_diag = -1 * np.ones(m - 1)  
    upper_diag = -1 * np.ones(m - 1) 
    
    tridiag_matrix = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csr')
    Phi = tridiag_matrix / (h**2)
        
    A = block_diag((kron(Phi, I) + kron(I, Phi), kron(Phi, I) + kron(I, Phi)))
    
    B = vstack([kron(I, Psi), kron(Psi, I)]).T
    
    Q = B@B.T
    
    m, n = B.shape
    
    x0 = np.zeros(n+m)
    f = np.ones(n)
    g = np.zeros(m)
    return A, B, Q, x0, f, g
    
    
np.random.seed(155)

def Ifiss_matrices(link):

    A = loadmat(link+'A.mat')["Ast"]
    B = loadmat(link+'B.mat')["Bst"][2:]
    Q = loadmat(link+'Q.mat')["Q"][2:, 2:]
    f = A@np.ones(A.shape[0]) + B.T@np.ones(B.shape[0]) 
    g = B@np.ones(A.shape[0]) 
    m, n = B.shape
    x0 = np.zeros(n + m)
    
    return A, B, Q, x0, f, g
    

# Preconditioning system solving Mz = v

def Gmres(A, f, x0, eps=1e-6,max_iter = 2000):
  n = len(f)
  H = np.zeros((max_iter+1,max_iter))
  x = np.zeros((max_iter, n))
  x[0] = x0
  v = np.zeros((max_iter, n))
  r = f - A@x[0]
  beta = norm(r)
  v[0] = r/beta
  k = 0
  r_k = norm(r)
  while r_k/norm(r) > eps and k < max_iter:
      w = np.zeros(n)
      w = A@v[k-1]
      for l in range(k):
          H[l,k-1] = v[l].T@w
          w -= H[l,k-1]*v[l]


      H[k,k-1] = norm(w)
      v[k] = w/H[k,k-1]
      e_1 = np.zeros(k+1)
      e_1[0] = 1
      H_k = H[:k+1,:k]
      y = QR_solve(H_k, beta * e_1) # QR
      x[k] = x[0] + v[:k].T@y
      r_k = f - A@x[k]
      r_k = norm(r_k)
      k += 1

  return x[k-1], k


def Solve(L, b): # using cholesky
    y = spsolve_triangular(L.T,b,lower=False)
    x = spsolve_triangular(L,y,lower=True)
    return x


def preconditioned_CG(A, L, f, max_iter=1000, eps=1e-6):
    m = len(f)
    u = np.random.rand(m)
    r = np.zeros((max_iter, m))
    z = np.zeros((max_iter, m))
    p = np.zeros((max_iter, m))
    alpha = np.zeros(max_iter)
    beta = np.zeros(max_iter)
    r[0] = f - A@u
    z[0] = L.solve(r[0])
    p[0] = z[0]
    k = 0
    while (k<max_iter-1) and (norm(r[k])/norm(r[0]) > eps):
        alpha[k] = (z[k].T@r[k]) / ((A@p[k]).T@p[k])
        u += alpha[k]*p[k]
        r[k+1] = r[k] - alpha[k]*(A@p[k])
        z[k+1] = L.solve(r[k+1])
        beta[k] = (z[k+1].T@r[k+1]) / (z[k].T@r[k])
        p[k+1] = z[k+1] + beta[k]*p[k]
        k += 1
    return u



# residual

def resid(x, A, B, f, g):
    m, n = B.shape
    return norm(np.concatenate((f - A@x[:n] - B.T@x[n:], g - B@x[:n])))


# realtive residual

def relative_resid(x, x0, A, B, f, g):
    m, n = B.shape
    return norm(np.concatenate((f - A@x[:n] - B.T@x[n:], g - B@x[:n])))/norm(np.concatenate((f - A@x0[:n] - B.T@x0[n:], g - B@x0[:n])))

# saddle GMRES

def QR_solve(A, b):
  q, r = qr(A)
  x = solve_triangular(r,q.T@b,lower=False) # QR
  return x

def saddle_GMRES(A, B, f, g, Q, x0, eps=1e-6,max_iter = 2000):
  start = time.time()
  m, n = B.shape
  H = np.zeros((max_iter+1,max_iter))
  x = np.zeros((n+m,max_iter))
  x[:,0] = x0
  v = np.zeros((n+m,max_iter))
  r = np.concatenate((f - A@x[:n,0] - B.T@x[n:,0],g - B@x[:n,0]))
  beta = norm(r)
  v[:,0] = r/beta
  k = 0
  r_k = norm(r)
  while r_k/norm(r) > eps and k < max_iter:
      w = np.zeros(n+m)
      w[:n] = A@v[:n,k-1] + B.T@v[n:,k-1]
      w[n:] = B@v[:n,k-1]
      for l in range(k):
          H[l,k-1] = v[:,l].T@w
          w -= H[l,k-1]*v[:,l]


      H[k,k-1] = norm(w)
      v[:,k] = w/H[k,k-1]
      e_1 = np.zeros(k+1)
      e_1[0] = 1
      H_k = H[:k+1,:k]
      y = QR_solve(H_k, beta * e_1) # QR
      x[:,k] = x[:,0] + v[:,:k]@y
      r_k = np.concatenate((f - A@x[:n,k] - B.T@x[n:,k],g - B@x[:n,k]))
      r_k = norm(r_k)
      k += 1

  return x[:,k-1], k, resid(x[:,k-1], A, B, f, g), relative_resid(x[:,k-1], x0, A, B, f, g), time.time() - start



def saddle_GMRES_restart(A, B, f, g, Q, x0, h=20, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0.copy()
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  beta = norm(r0)
  beta0 = beta.copy()
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w = np.concatenate((A@v[j,:n] + B.T@v[j, n:], B@v[j, :n]))
        
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r = np.concatenate((f - A@x[:n] - B.T@x[n:], g - B@x[:n]))
      beta = norm(r)
      print(beta/beta0)
      if (beta/beta0) < eps or (k >= max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start

# Diagonal Preconditioned GMRES

def Diagonal_prefunction(A, Q, v_1, v_2):
    n = len(v_1)
    m = len(v_2)
    z_1, _ = cg(A, v_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2, _ = cg(Q, v_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    return z_1, z_2


def saddle_preconditioned_GMRES_restart(A, B, f, g, Q, x0, h=10, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0.copy()
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = Diagonal_prefunction(A, Q, r0[:n], r0[n:])
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = Diagonal_prefunction(A, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n])
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r[:n], r[n:] = Diagonal_prefunction(A, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n])
      beta = norm(r)
      if (relative_resid(x, x_init, A, B, f, g)) < eps or (k >= max_iter):
          convergence = True
      else:
        x_init = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start
# Triangular Preconditioned GMRES


def Triangular_prefunction(A, B, Q, v_1, v_2):
    n = len(v_1)
    m = len(v_2)
    z_2, _ = cg(Q, v_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1, _ = cg(A, v_1-B.T@z_2, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)

    return z_1, z_2

def saddle_preconditioned_GMRES_restart_Triangular(A, B, f, g, Q, x0, h=100, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0.copy()
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = Triangular_prefunction(A, B, Q, r0[:n], r0[n:]) # preconditioning
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = Triangular_prefunction(A, B, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n]) # preconditioning
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r[:n], r[n:] = Triangular_prefunction(A, B, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n]) # preconditioning
      beta = norm(r)
      if (relative_resid(x, x_init, A, B, f, g)) < eps or (k >= max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start

# Regularized Preconditioned GMRES

def regularized_preconditioner(A, B, Q, v_1, v_2, alpha):

    m, n = B.shape
    x_init_1 = np.random.rand(n)
    x_init_2 = np.random.rand(m)
    # P_1@y = v

    y_1 = v_1
    y, _ = cg(A, v_1, x0 = x_init_1, tol=1E-6, maxiter=1000)
    y_2 = v_2 - B@y

    # P_2@x = y
    x_1, _ = cg(A, y_1, x0 = x_init_1, tol=1E-6, maxiter=1000)
    x_2, _ = cg(Q, y_2/(alpha-1), x0 = x_init_2, tol=1E-6, maxiter=1000)

    # P_3@z = x

    z_1, _ = cg(A, A@x_1 - B.T@x_2, x0 = x_init_1, tol=1E-6, maxiter=1000)
    z_2 = x_2


    return z_1, z_2


# GMRHSS


def GMRHSS_preconditioner(A, B, Q, v_1, v_2, alpha, beta):
    n = len(v_1)
    m = len(v_2)
    t, _ = cg(A, v_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2, _ = cg(beta*Q + Q + B@B.T/alpha, v_2 + B@t, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1 = t - B.T@z_2/alpha
    
    return z_1, z_2


def GMRHSS(A, B, f, g, Q, x0, h=100, alpha=0.1, beta=0.5, tol=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  Q = scipy.sparse.diags((B@B.T).diagonal())
  x_init = x0.copy()
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  print(len(r0[:n]))
  r0[:n], r0[n:] = GMRHSS_preconditioner(A, B, Q, alpha, beta, r0[:n], r0[n:])
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = GMRHSS_preconditioner(A, B, Q, alpha, beta, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n])
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r[:n], r[n:] = GMRHSS_preconditioner(A, B, Q, alpha, beta, f - A@x[:n] - B.T@x[n:], g - B@x[:n])
      beta = norm(r)
      print(relative_resid(x, x_init, A, B, f, g))
      if (relative_resid(x, x_init, A, B, f, g)) < tol or (k >= max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start


def GMRES(preconditioner, A, B, f, g, Q, x0 , h=100, tol=1e-6,max_iter = 1000, *args, **kwargs):
  start = time.time()
  #Q = scipy.sparse.diags(Q.diagonal())
  m, n = B.shape
  x_init = x0.copy()
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = preconditioner(A, B, Q, r0[:n], r0[n:], *args, **kwargs)
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = preconditioner(A, B, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n], *args, **kwargs)
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r[:n], r[n:] = preconditioner(A, B, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n], *args, **kwargs)
      beta = norm(r)
      print(relative_resid(x, x_init, A, B, f, g))
      if (relative_resid(x, x_init, A, B, f, g)) < tol or (k >= max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start


def restarted_GMRES(preconditioner, A, B, f, g, Q, x0 , h=100, tol=1e-6,max_iter = 1000, *args, **kwargs):
    
  start = time.time()
  m, n = B.shape
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = preconditioner(A, B, Q, r0[:n], r0[n:], *args, **kwargs)
  beta0 = beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = preconditioner(A, B, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n], *args, **kwargs)
          for i in range(j):
              H[i,j] = v[i].T@w
              w -= H[i,j]*v[i]


          H[j+1,j] = norm(w)
          v[j+1] = w/H[j+1,j]

      V_h = v[:h]
      e_1 = np.zeros(h+1)
      e_1[0] = 1
      y = QR_solve(H, beta * e_1) # QR
      x = x0 + V_h.T@y
      r = np.zeros(n+m)
      r[:n], r[n:] = preconditioner(A, B, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n], *args, **kwargs)
      beta = norm(r)
      print(beta/beta0)
      if (beta/beta0 < tol) or (k >= max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, beta, beta/beta0, time.time() - start

# Non Symetric Saddle Preconditionned GMRES

def NonSymetric_Regularized_GMRES(A, B, f, g, Q, x0, alpha, max_iter=1000, tol=1e-6):
    B = B.T
    n, m = B.shape
    q = np.array([Q[i,i] for i in range(m)])
    Q_inv = scipy.sparse.diags(1/q)
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B@x[n:]
    r2 = -g + B.T@x[:n]
    beta0 = norm(np.concatenate((r1, r2)))
    A_alpha = A + (2/alpha)*B@Q_inv@B.T
    k = 0
    while not convergence:
        rhs = r1 - (2/alpha) * B@Q_inv@r2
        z1, _ = cg(A_alpha, rhs, x0=x0[:n],tol=1E-6, maxiter=1000)
        #z1, _ = GMRES(A_alpha, rhs, x0=x0[:n])
        z2 = (1/alpha) * Q_inv@(r2 + B.T@z1)
        x += np.concatenate((z1, z2))
        r1 = f - A@x[:n] - B@x[n:]
        r2 = -g + B.T@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        

    return x, k, beta/beta0


def preconditioner(A, B, Q, v_1, v_2, alpha, beta):
    z_1 = v_1
    m = len(v_2)
    I = csc_matrix(np.identity(m))
    z_2, _ = cg(beta*I + Q, alpha*v_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    return z_1, z_2
    

# Shift-splitting preconditioner

def SS(A, B, f, g, Q, x0, alpha=0.2, max_iter=1000, tol=1e-6): 
    m, n = B.shape
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    I = csc_matrix(np.identity(n))
    A_alpha = alpha*I - A 
    B_alpha = alpha*I + A + B.T@B/alpha
    
    while not convergence:
        rhs1 = f + (A_alpha@x[:n] - B.T@x[n:])/2
        rhs2 = -g + (B@x[:n] + alpha*x[n:])/2
        
        
        # step 1

        t1 = rhs1 - B.T@rhs2/alpha

        # step 2
        
        x[:n], _ = cg(B_alpha, 2*t1, x0=x0[:n],tol=1E-6, maxiter=1000)
        
        
        # step 3

        x[n:] = (B@x[:n] + 2*rhs2)/alpha
        
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0

def SS_preconditioner(A, B, Q, v_1, v_2, alpha): # alpha = 0.2, iter = 9

    n = len(v_1)
    I = csc_matrix(np.identity(n))
    z_1, _ = cg(alpha*I + A + B.T@B/alpha, 2*(v_1-B.T@v_2/alpha), x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = (B@z_1 + 2*v_2)/alpha
    
    return z_1, z_2


# Local shift-splitting preconditioner

def LSS(A, B, f, g, Q, x0, alpha=0.2, max_iter=1000, tol=1e-6): # stagnate
    m, n = B.shape
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0 
    B_alpha = A + B.T@B/alpha
    
    while not convergence:
        rhs1 = f - (A@x[:n] + B.T@x[n:])/2
        rhs2 = -g + (B@x[:n] + alpha*x[n:])/2
        
        
        # step 1

        t1 = rhs1 - B.T@rhs2/alpha

        # step 2
        
        x[:n], _ = cg(B_alpha, 2*t1, x0=x0[:n],tol=1E-6, maxiter=1000)
        
        # step 3

        
        x[n:] = (B@x[:n] + 2*rhs2)/alpha
        
        
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0

def LSS_preconditioner(A, B, Q, v_1, v_2, alpha): # alpha = 0.2, iter = 14
    n = len(v_1)
    z_1, _ = cg(A + B.T@B/alpha, 2*(v_1-B.T@v_2/alpha), x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = (B@z_1 + 2*v_2)/alpha
    
    return z_1, z_2



# Generalized Shift-Splitting Preconditioner 

def GSS_preconditioner(A, B, Q, v_1, v_2, alpha, beta): # alpha=0.1, beta=0.2, iter = 10
    n = len(v_1)
    I = csc_matrix(np.identity(n))
    t = v_1 - (1/beta)*B.T@v_2
    z_1, _ = cg(alpha*I + A + B.T@B/beta, 2*t, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = (B@z_1 + 2*v_2)/beta
    
    return z_1, z_2


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
    while not convergence:
        
        # solve first system
        
        rhs1 = alpha * x[:n] - E@x[n:] + f
        rhs2 = E.T@x[:n] + alpha*x[n:] - g
        #z1, _ = preconditioned_CG(A_alpha, L_A ,rhs1, max_iter=1000, eps=1e-6)
        z1, _ = cg(A_alpha, rhs1, x0=x0[:n],tol=1E-6, maxiter=1000)
        z2 = rhs2/alpha
        
        # solve second system

        rhs1 = f + A_alpha_@z1
        rhs2 = -g + alpha*z2
        
       # step1 
       
        t1 = rhs1 - E@rhs2/alpha
       
       # step2
        
        #x[:n], _ = preconditioned_CG(E_alpha, L_E, 2*t1, x0=x0[:n],tol=1E-6, maxiter=1000)
        x[:n], _ = cg(E_alpha, t1, x0=x0[:n],tol=1E-6, maxiter=1000)
       
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
    v_1, _ = cg(alpha*I + A, r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    v_2 = r_2/alpha
    t = v_1 - (1/alpha)*B.T@v_2
    z_1, _ = cg(alpha*I + B.T@B/alpha, t, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = (B@z_1 + v_2)/alpha
    
    return z_1, z_2



# Inexact Iteration Hermitian and skew-Hermitian splitting method

def IHSS(A, B, f, g, Q, x0, alpha, max_iter=1000, tol=1e-6): 
    m, n = B.shape
    I = csc_matrix(np.identity(n))
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n]
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    A_alpha = alpha*I + A

    while not convergence:
        
        # solve first system

        z1, _ = cg(A, r1, x0=x0[:n],tol=1E-6, maxiter=1000)

        x[:n] += z1
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n]
        
        # solve second system
        
        z1, _ = cg(A_alpha, r1, x0=x0[:n],tol=1E-6, maxiter=1000)
        z2 = r2/alpha
        
        x[:n] += z1
        x[n:] += z2
        
       

        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        
    
    return x, k, beta/beta0


# Regularized Hermitian and skew-Hermitian splitting method

def RHSS(A, B, f, g, Q, x0, alpha=0.2, max_iter=1000, tol=1e-6): # cv in 441
    B = B.T
    n, m = B.shape
    I = csc_matrix(np.identity(max(n,m)))
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B@x[n:]
    r2 = -g + B.T@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    BtB = B.T@B
    A_alpha = alpha*I[:n,:n] + A
    A_alpha_ = alpha*I[:n,:n] - A
    Q_alpha = alpha*I[:m,:m] + Q
    while not convergence:
        
        # solve first system
        
        rhs1 = alpha * x[:n] - B@x[n:] + f
        rhs2 = B.T@x[:n] + Q_alpha@x[n:] + g
        z1, _ = cg(A_alpha, rhs1, x0=x0[:n],tol=1E-6, maxiter=1000)
        # solve second system

        rhs1 = f + (A_alpha_)@z1
        rhs2 = 2*g + B.T@x[:n] + (Q_alpha)@x[n:]
        x[n:], _ = cg(Q_alpha + (1/alpha)*BtB, rhs2 + (1/alpha)*B.T@rhs1, x0=x0[n:],tol=1E-6, maxiter=1000)
        x[:n] = (1/alpha) * (rhs1 - B@x[n:])
        
        r1 = f - A@x[:n] - B@x[n:]
        r2 = -g + B.T@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        
    
    return x, k, beta/beta0


def RHSS_preconditioner(A, B, Q, r_1, r_2, alpha):

    n = len(r_1)
    m = len(r_2)
    I = csc_matrix(np.identity(max(n, m)))
    
    x_1, _ = cg(alpha*I[:n,:n] + A, 2*alpha*r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    x_2 = 2*r_2

    #z_2, _ = cg(B@B.T, alpha*(B@x_1/alpha + x_2)/2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_2, _ = cg(alpha*I[:m,:m] + 2*B@B.T/alpha, B@x_1/alpha + x_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1 = (x_1 - B.T@z_2)/alpha
    
    return z_1, z_2

def RHSS_preconditioner1(A, B, Q, r_1, r_2, alpha): # alpha=1, iter = 59
    
    n = len(r_1)
    m = len(r_2)
    I = csc_matrix(np.identity(max(n, m)))
    v_1, _ = cg(alpha*I[:n,:n] + A, alpha*r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    v_2 = r_2
    
    z_2, _ = cg(alpha*I[:m,:m] + Q + B@B.T/alpha, B@v_1/alpha + v_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1 = (v_1 - B.T@z_2)/alpha
    
    return z_1, z_2


# Accelerated Regularized Hermitian and skew-Hermitian splitting preconditioner

def ARHSS(A, B, f, g, Q, x0, alpha, beta, tol=1e-6,max_iter = 1000): #stagnate
    E = B.T
    m, n = B.shape
    I = csc_matrix(np.identity(max(n, m)))
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n]
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    A_alpha = alpha*I[:n,:n] + A
    A_alpha_ = alpha*I[:n,:n] - A
    Q_alpha = alpha*I[:m,:m] + Q
    E_alpha = beta*I[:m,:m] + Q + E.T@E/alpha
    while not convergence:
        
        # step 1
        
        rhs1 = alpha * x[:n] - E@x[n:] + f
        z1, _ = cg(A_alpha, rhs1, x0=x0[:n],tol=1E-6, maxiter=1000)
 
        
        # step 2

        rhs1 = f + A_alpha_@z1
        rhs2 = 2*g + E.T@x[:n] + Q_alpha@x[n:]
        
       # step 3
       

        
        x[n:], _ = cg(E_alpha, 0.2*E.T@rhs1 + rhs2, x0=x0[n:],tol=1E-6, maxiter=1000)
       
       # step 4
       
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        
    
    return x, k, beta/beta0

def ARHSS_preconditioner(A, B, Q, r_1, r_2, alpha, beta): # 10, 15, iter = 106

    n = len(r_1)
    m = len(r_2)
    I = csc_matrix(np.identity(max(n, m)))
    
    x_1, _ = cg(alpha*I[:n,:n] + A, 2*alpha*r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    x_2 = 2*r_2


    z_2, _ = cg(beta*I[:m,:m] + Q + B@B.T/alpha, B@x_1/alpha + x_2, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1 = (x_1 - B.T@z_2)/alpha
    
    return z_1, z_2


# Preconditioned Hermitian and skew-Hermitian splitting

def PHSS(A, B, f, g, Q, x0, alpha=1, max_iter=1000, tol=1e-6): # cv in 9 iter
    m, n = B.shape
    E = B.T
    C = np.array([Q[i,i] for i in range(m)])
    C_inv = scipy.sparse.diags(1/C)
    C = scipy.sparse.diags(C)
    AECE = alpha*A + E@C_inv@E.T/alpha
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0

    
    while not convergence:
        
        rhs1 = alpha*(alpha-1)/(alpha+1) * A@x[:n] - (alpha-1)/(alpha+1) * E@x[n:] + 2*alpha/(alpha+1)*f
        rhs2 = E.T@x[:n] + alpha*Q@x[n:] + 2*g
 

        # P_1@u = rhs
        
        u2 = rhs2
        u3, _ = cg(C, rhs2, x0=x0[n:],tol=1E-6, maxiter=1000)
        u1 = rhs1 - E@u3/alpha


        # P_2@v = u
        
  
        v1, _ = cg(AECE, u1, x0=x0[:n],tol=1E-6, maxiter=1000)
        v2, _ = cg(C, u2, x0=x0[n:]/alpha,tol=1E-6, maxiter=1000)
        
        # P_3@x = v

        x[:n] = v1
        x3, _ = cg(C, E.T@x[:n], x0=x0[n:],tol=1E-6, maxiter=1000)
        x[n:]  = v2 + x3/alpha
        
 
        
 
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0





def PHSS_preconditioner(A, B, Q, r_1, r_2, alpha): #  alpha  = 1, iter = 46

    n = len(r_1)
    m = len(r_2)
    
    C = np.array([Q[i,i] for i in range(m)])
    C_inv = scipy.sparse.diags(1/C)
    
    t = r_1 - (1/alpha)*B.T@C_inv@r_2
    z_1, _ = cg(alpha*A + B.T@C_inv@B/alpha, t, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = C_inv@(B@z_1 + r_2)/alpha
    
    return z_1, z_2

# Accelerated Hermitian and skew-Hermitian splitting preconditioner

def AHSS(A, B, f, g, Q, x0, alpha, beta, tol=1e-6,max_iter = 1000): #dv 
    m, n = B.shape
    E = B.T
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    #ETE = E.T@E
    #EET = E@E.T
    k = 0

    
    while not convergence:
        
        
        # auxiliary vector
        u = 2*r1/(1+alpha)
        y, _ = cg(A, u, x0=x0[:n],tol=1E-6, maxiter=1000)
        v = E.T@y + 2*r2
        
        
        # update vector
      
        
        #z, _ = cg((beta+1/alpha)*EET, E@v, x0=x0[:n],tol=1E-6, maxiter=1000)
        
        
        #w, _ = cg(ETE, E.T@A@z, x0=x0[n:],tol=1E-6, maxiter=1000)
        
        w, _ = cg(Q, v, x0=x0[n:]/(beta+1/alpha),tol=1E-6, maxiter=1000)
        
        #w, _ = cg(beta*ETE +1/alpha*Q, v, x0=x0[n:],tol=1E-6, maxiter=1000)
        
        t, _ = cg(A, u - E@w, x0=x0[:n],tol=1E-6, maxiter=1000)
        
        # next iterate
        
        x[:n] += t
        x[n:] += w
        
        

 
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0


def AHSS_preconditioner1(A, B, Q, r_1, r_2, alpha, beta): # alpha = 5, beta = 10, iter = 136

    n = len(r_1)
    m = len(r_2)
    
    C = np.array([Q[i,i] for i in range(m)])
    C_inv = scipy.sparse.diags(1/C)
    
    t = r_1 - (1/beta)*B.T@C_inv@r_2
    z_1, _ = cg(alpha*A + B.T@C_inv@B/beta, t, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2 = C_inv@(B@z_1 + r_2)/beta
    
    return z_1, z_2

def AHSS_preconditioner(A, B, Q, r_1, r_2, alpha, beta): # alpha = 5, beta = 10, iter = 19

    n = len(r_1)
    m = len(r_2)

    v_1, _ = cg(A, r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    t = (r_2 + (1/alpha)*B@v_1)/(beta + 1/alpha)
    z_2, _ = cg(Q, t, x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    z_1,  _ = cg(A, (r_1-B.T@z_2)/alpha, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    
    return z_1, z_2


# Generalized successive overrelaxation

def GSOR(A, B, f, g, Q, x0, w=0.91, tau=0.91, tol=1e-6,max_iter = 1000):
    m, n = B.shape
    E = B.T
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B.T@x[n:]
    r2 = g - B@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0

    
    while not convergence:
        
        rhs1 = (1-w)*A@x[:n] + w*(f-E@x[n:])
        x[:n], _ = cg(A, rhs1, x0=x0[:n],tol=1E-6, maxiter=1000)
        rhs2 = Q@x[n:] + tau*(E.T@x[:n] - g)
        x[n:], _ = cg(Q, rhs2, x0=x0[n:],tol=1E-6, maxiter=1000)
        
        

        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0

def GSOR_preconditioner(A, B, Q, r_1, r_2, w, tau): # w=0.1, tau=0.05, iter = 16
    n = len(r_1)
    m = len(r_2)
    z_1, _ = cg(A, w*r_1, x0 = np.random.rand(n) ,tol=1E-6, maxiter=1000)
    z_2, _ = cg(Q, tau*(r_2 + B@z_1), x0 = np.random.rand(m) ,tol=1E-6, maxiter=1000)
    return z_1, z_2


# Saddle MINRES

def saddle_MINRES(A, B, f, g, Q, x0, max_iter=2000, eps=1e-6):
    start = time.time()
    m, n = B.shape
    x = np.zeros((n+m,max_iter))
    v = np.zeros((n+m,max_iter+1))
    w = np.zeros((n+m,max_iter+1))
    s = np.zeros(max_iter+1)
    c = np.zeros(max_iter+1)
    gamma = np.zeros(max_iter+1)
    delta = np.zeros(max_iter+1)
    x[:,0] = x0
    v[:n,1] = f - A@x[:n,0] - B.T@x[n:,0]
    v[n:,1] = g - B@x[:n,0]
    gamma[1] = norm(v[:,1])
    eta = gamma[1]
    s[0] = s[1] = 0
    c[0] = c[1] = 1
    j = 1
    while (j < max_iter-1) and norm(np.concatenate((f - A@x[:n,j-1] - B.T@x[n:,j-1], g - B@x[:n,j-1])))/norm(v[:,1]) > eps :
        v[:,j] /= gamma[j]
        delta[j] = (np.concatenate((A@v[:n,j]+B.T@v[n:,j],B@v[:n,j]))).T@v[:,j]
        v[:,j+1] = np.concatenate((A@v[:n,j]+B.T@v[n:,j] , B@v[:n,j])) - delta[j]*v[:,j] - gamma[j]*v[:,j-1]
        gamma[j+1] = norm(v[:,j+1])
        alpha_0 = c[j]*delta[j] - c[j-1]*s[j]*gamma[j]
        alpha_1 = np.sqrt(alpha_0**2 + gamma[j+1]**2)
        alpha_2 = s[j]*delta[j] + c[j-1]*c[j]*gamma[j]
        alpha_3 = s[j-1]*gamma[j]
        c[j+1] = alpha_0/alpha_1
        s[j+1] = gamma[j+1]/alpha_1
        w[:,j+1] = (v[:,j] - alpha_3*w[:,j-1] - alpha_2*w[:,j])/alpha_1
        x[:,j] = x[:,j-1] + c[j+1]*eta*w[:,j+1]
        eta = -s[j+1]*eta
        j += 1

    return x[:,j-1], j, resid(x[:,j-1], A, B, f, g), relative_resid(x[:,j-1], x0, A, B, f, g), time.time() - start



# Preconditioned MINRES


def preconditioned_MINRES(preconditioner, A, B, f, g, Q, x0, max_iter=1000, eps=1e-6, *args, **kwargs): # page 207
    start = time.time()
    n = len(f)
    m = len(g)
    x = np.zeros((n+m,max_iter))
    v = np.zeros((n+m,max_iter+1))
    w = np.zeros((n+m,max_iter+1))
    z = np.zeros((n+m,max_iter+1))
    s = np.zeros(max_iter+1)
    c = np.zeros(max_iter+1)
    gamma = np.zeros(max_iter+1)
    gamma[0] = 1 # /0 ?
    delta = np.zeros(max_iter+1)
    x[:,0] = x0
    v[:n,1] = f - A@x[:n,0] - B.T@x[n:,0]
    v[n:,1] = g - B@x[:n,0]
    z[:n,1], z[n:,1] = preconditioner(A, B, Q, v[:n,1], v[n:,1], *args, **kwargs) # solve Mz = v
    gamma[1] = np.sqrt(v[:,1].T@z[:,1])
    eta = gamma[1]
    s[0] = s[1] = 0
    c[0] = c[1] = 1
    j = 1
    while (j <= max_iter-1) and norm(np.concatenate((f - A@x[:n,j-1] - B.T@x[n:,j-1], g - B@x[:n,j-1])))/norm(v[:,1]) > eps :
        z[:,j] /= gamma[j]
        delta[j] = (np.concatenate((A@z[:n,j] + B.T@z[n:,j], B@z[:n,j]))).T@z[:,j]
        v[:,j+1] = np.concatenate((A@z[:n,j] + B.T@z[n:,j], B@z[:n,j])) - (delta[j]/gamma[j])*v[:,j] - (gamma[j]/gamma[j-1])*v[:,j-1]
        z[:n,j+1], z[n:,j+1] = preconditioner(A, B, Q, v[:n,j+1], v[n:,j+1], *args, **kwargs) # solve Mz = v
        gamma[j+1] = np.sqrt(z[:,j+1].T@v[:,j+1])
        alpha_0 = c[j]*delta[j] - c[j-1]*s[j]*gamma[j]
        alpha_1 = np.sqrt(alpha_0**2 + gamma[j+1]**2)
        alpha_2 = s[j]*delta[j] + c[j-1]*c[j]*gamma[j]
        alpha_3 = s[j-1]*gamma[j]
        c[j+1] = alpha_0/alpha_1
        s[j+1] = gamma[j+1]/alpha_1
        w[:,j+1] = (z[:,j] - alpha_3*w[:,j-1] - alpha_2*w[:,j])/alpha_1
        x[:,j] = x[:,j-1] + c[j+1]*eta*w[:,j+1]
        eta = -s[j+1]*eta
        j += 1
    return x[:,j-1], j, resid(x[:,j-1], A, B, f, g), relative_resid(x[:,j-1], x0, A, B, f, g), time.time() - start