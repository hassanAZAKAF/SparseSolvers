Features:
  Krylov Subspace Methods: Implementation of GMRES, Conjugate Gradient (CG), and MINRES algorithms.
  Preconditioning Techniques: Integration of preconditioners to accelerate solver convergence.

Files in the Repository
  GlobalSolvers.py: Implements global solvers for handling Stokes systems with multiple right-hand sides.
  GlobalSolversNavierStokes.py: Specialized solvers for tackling the nonlinear Navier-Stokes equations with multiple right-hand sides.
  NavierStokesSolvers.py: Contains methods and preconditioning strategies specifically for Navier-Stokes systems with one right-hand side.
  StokesSolvers.py: Focused on the linear Stokes problem, with dedicated solvers and preconditioners for one right-hand side.

Prerequisites
Python 3.10
NumPy
SciPy
PyMUMPS
