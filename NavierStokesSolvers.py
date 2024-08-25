# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:42:56 2024

@author: allo
"""

# Importing Libraries
import numpy as np
from numpy.linalg import norm, qr
import scipy
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import  spsolve_triangular
from scipy.sparse import csc_matrix
from mumps import DMumpsContext
import time


np.random.seed(155)

# Preconditioning system solving Mz = v

def Solve(L, b): # using cholesky
    y = spsolve_triangular(L.T,b,lower=False)
    x = spsolve_triangular(L,y,lower=True)
    return x


def Mumps_solver(A, f):
    ctx = DMumpsContext()
    ctx.set_icntl(35, 2) # Activate Block Low Rank Factorization
    if ctx.myid == 0:
        ctx.set_centralized_sparse(A)
        x = f.copy()
        ctx.set_rhs(x) # Modified in place
        
    ctx.set_silent()
    ctx.run(job=6) # Analysis + Factorization + Solve

    ctx.destroy() # Free memory
    return x 


def preconditioned_CG(A, L, f, max_iter=1000, eps=1e-6):
    m = len(f)
    u = np.zeros((m,max_iter))
    r = np.zeros((m,max_iter))
    z = np.zeros((m,max_iter))
    p = np.zeros((m,max_iter))
    u[:,0] = np.random.rand(m)
    alpha = np.zeros(max_iter)
    beta = np.zeros(max_iter)
    r[:,0] = f - A@u[:,0]
    z[:,0] = Solve(L, r[:,0])
    p[:,0] = z[:,0]
    k = 0
    while (k<max_iter-1) and (norm(r[:,k])/norm(r[:,0]) > eps):
        alpha[k] = (z[:,k].T@r[:,k]) / ((A@p[:,k]).T@p[:,k])
        u[:,k+1] = u[:,k] + alpha[k]*p[:,k]
        r[:,k+1] = r[:,k] - alpha[k]*(A@p[:,k])
        z[:,k+1] = Solve(L, r[:,k+1])
        beta[k] = (z[:,k+1].T@r[:,k+1]) / (z[:,k].T@r[:,k])
        p[:,k+1] = z[:,k+1] + beta[k]*p[:,k]
        k += 1
    return u[:,k-1], k

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

# Diagonal Preconditioned GMRES

def Diagonal_Mumps_prefunction(A, Q, v_1, v_2):
    z_1 = Mumps_solver(A, v_1)
    z_2 = Mumps_solver(Q, v_2)
    return z_1, z_2

def saddle_preconditioned_GMRES_restart(A, B, f, g, Q, x0, h=100, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = Diagonal_Mumps_prefunction(A, Q, r0[:n], r0[n:])
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = Diagonal_Mumps_prefunction(A, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n])
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
      r[:n], r[n:] = Diagonal_Mumps_prefunction(A, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n])
      beta = norm(r)
      if (relative_resid(x, x_init, A, B, f, g)) < eps or (k > max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start
# Triangular Preconditioned GMRES



def Triangular_Mumps_prefunction(A, B, Q, v_1, v_2):
    z_2 = Mumps_solver(Q, v_2)
    z_1 = Mumps_solver(A, v_1-B.T@z_2)
    return z_1, z_2


def saddle_preconditioned_GMRES_restart_Triangular(A, B, f, g, Q, x0, h=100, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = Triangular_Mumps_prefunction(A, B, Q, r0[:n], r0[n:]) # preconditioning
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = Triangular_Mumps_prefunction(A, B, Q, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n]) # preconditioning
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
      r[:n], r[n:] = Triangular_Mumps_prefunction(A, B, Q, f - A@x[:n] - B.T@x[n:], g - B@x[:n]) # preconditioning
      beta = norm(r)
      if (relative_resid(x, x_init, A, B, f, g)) < eps or (k > max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start

# Regularized Preconditioned GMRES


def regularized_Mumps_prefunction(A, B, Q, alpha, v_1, v_2):
    y_1 = v_1
    y = Mumps_solver(A, v_1)
    y_2 = v_2 - B@y

    # P_2@x = y
    x_1 = Mumps_solver(A, y_1)
    x_2 = Mumps_solver(Q, y_2/(alpha-1))

    # P_3@z = x

    z_1 = Mumps_solver(A, A@x_1 - B.T@x_2)
    z_2 = x_2

    return z_1, z_2

def saddle_preconditioned_GMRES_restart_Regularized(A, B, f, g, Q, x0, alpha=10, h=100, eps=1e-6,max_iter = 1000):
  start = time.time()
  m, n = B.shape
  x_init = x0
  v = np.zeros((max_iter+1, n+m))
  r0 = np.concatenate((f - A@x0[:n] - B.T@x0[n:],g - B@x0[:n]))
  r0[:n], r0[n:] = regularized_Mumps_prefunction(A, B, Q, alpha,  r0[:n], r0[n:]) # preconditioning
  beta = norm(r0)
  convergence = False
  k = 0
  while not convergence:
      v[0] = r0 / beta
      w = np.zeros(n+m)
      H = np.zeros((h+1,h))
      for j in range(h):
          w[:n], w[n:] = regularized_Mumps_prefunction(A, B, Q, alpha, A@v[j,:n] + B.T@v[j, n:], B@v[j, :n]) # preconditioning
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
      r[:n], r[n:] = regularized_Mumps_prefunction(A, B, Q, alpha, f - A@x[:n] - B.T@x[n:], g - B@x[:n]) # preconditioning
      beta = norm(r)
      if (relative_resid(x, x_init, A, B, f, g)) < eps or (k > max_iter):
          convergence = True
      else:
        x0 = x
        r0 = r
        k = k+1

  return x, k, resid(x, A, B, f, g), relative_resid(x, x_init, A, B, f, g), time.time() - start


# New Preconditioners

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
        z1 = Mumps_solver(A_alpha, rhs)
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
        
        x[:n] = Mumps_solver(B_alpha, 2*t1)
        
        
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
        
        x[:n] = Mumps_solver(B_alpha, 2*t1)
        
        
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


# Hermitian and skew-Hermitian splitting method

def HSS(A, B, f, g, Q, x0, alpha, max_iter=1000, tol=1e-6): # dv
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
        z1 = Mumps_solver(A_alpha, rhs1)
        z2 = (E.T@x[:n] + alpha*x[n:] - g)/alpha
        
        # solve second system

        rhs1 = f + (A_alpha_)@z1
        rhs2 = -g + alpha*z2
        
       # step1 
       
        t1 = rhs1 - E@rhs2/alpha
       
       # step2
        
        x[:n] = Mumps_solver(E_alpha, 2*t1)
       
       # step3
       
        x[n:] = (E.T@x[:n] + 2*rhs2)/alpha
        
        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n]
        beta = norm(np.concatenate((r1, r2)))
        print(beta/beta0)
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
        
    
    return x, k, beta/beta0


# regularized Hermitian and skew-Hermitian splitting method

def RHSS(A, B, f, g, Q, x0, alpha=0.2, max_iter=1000, tol=1e-6):
    B = B.T
    n, m = B.shape
    I = csc_matrix(np.identity(n + m))
    convergence = False
    x = x0.copy()
    r1 = f - A@x[:n] - B@x[n:]
    r2 = -g + B.T@x[:n] 
    beta0 = norm(np.concatenate((r1, r2)))
    k = 0
    BtB = B.T@B
    A_alpha = alpha*I[:n,:n] + A
    A_alpha_ = alpha*I[:n,:n] - A
    Q_alpha = alpha*I[n:,n:] + Q
    while not convergence:
        
        # solve first system
        
        rhs1 = alpha * x[:n] - B@x[n:] + f
        rhs2 = B.T@x[:n] + Q_alpha@x[n:] + g
        z1 = Mumps_solver(A_alpha, rhs1)
        
        # solve second system

        rhs1 = f + (A_alpha_)@z1
        rhs2 = 2*g + B.T@x[:n] + (Q_alpha)@x[n:]
        x[n:] = Mumps_solver(Q_alpha + (1/alpha)*BtB, rhs2 + (1/alpha)*B.T@rhs1)
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
        u3 = Mumps_solver(C, rhs2)
        u1 = rhs1 - E@u3/alpha


        # P_2@v = u
        
  
        v1 = Mumps_solver(AECE, u1)
        v2 = Mumps_solver(alpha*C, u2)
        
        # P_3@x = v

        x[:n] = v1
        x3= Mumps_solver(C, E.T@x[:n])
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



# Accelerated Hermitian and skew-Hermitian splitting preconditioner

def AHSS(A, B, f, g, Q, x0, alpha, beta, tol=1e-6,max_iter = 1000):
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
        y = Mumps_solver(A, u)
        v = E.T@y + 2*r2
        
        
        # update vector
      
        
        w = Mumps_solver((beta+1/alpha)*Q, v)
        
        t = Mumps_solver(A, u - E@w)
        
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
        x[:n] = Mumps_solver(A, rhs1)
        rhs2 = Q@x[n:] + tau*(E.T@x[:n] - g)
        x[n:] = Mumps_solver(Q, rhs2)
        
        

        r1 = f - A@x[:n] - B.T@x[n:]
        r2 = g - B@x[:n] 
        beta = norm(np.concatenate((r1, r2)))
        if (beta/beta0) < tol or (k >= max_iter):
            convergence = True
        else:
          k = k+1
    
    return x, k, beta/beta0

