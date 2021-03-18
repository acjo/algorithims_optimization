# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Caelan Osman
Math 323 Sec. 2
March 16, 2021
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m, n = A.shape
    #n x n zero matrix
    zero_nn = np.zeros((n, n))
    #m x n zero matrix
    zero_mn = np.zeros((m, n))
    #m x m zero matrix
    zero_mm = np.zeros((m, m))
    #n x n identity amtrix
    I_nn = np.eye(n)
    block_1 = np.column_stack((zero_nn, A.T, I_nn))
    block_2 = np.column_stack((A, zero_mm, zero_mn))

    block_1_2 = np.vstack((block_1, block_2))

    def F(x, lamb, mu):
        ''' defines the vector vaued function used the interior point problem
            Paramaters:
                x ((n,) ndarray): variable in objective function
                lambda ((m, ) ndarray): KKT condition lagrange multiplier
                mu ((n, ) ndarray): KKT condition lagrange multiplier
            Returns:
                ((2n +m, ) ndaraay) output of function
        '''
        return np.array([A.T @ lamb + mu - c, A @ x - b, np.diag(mu) @ x])

    def block3(x, mu):
        '''Returns the last block row in the block matrix defining the
           Ferchet derivative.
            Paramaters:
                x ((n,) ndarray): variable in objective function
                mu ((n, ) ndarray): KKT condition lagrange multiplier
            Returns:
                ((n,n2n+m) ndaraay) output of function

        '''
        M = np.diag(mu)
        X = np.diag(x)
        m, n = A.shape
        #n x m zero matrix
        zero_nm = np.zeros((n, m))
        #return bottom block
        return np.column_stack((M, zero_nm, X))

    #search direction subtroutine
    def direction(x, lamb, mu, nu, sigma=0.1):
        '''searching direction subroutine
           Paramaters:
               sigma (float) defailting to 0.1 used for scaling
            Returns:
                delta_x ((n, ) ndarray) delta x
                delta_lambda ((m, ) ndarray) delta lambda
                delta_mu ((n, ) ndarray) delta mu
        '''
        block_3 = block3(x, mu)
        DF = np.vstack((block_1_2, block_3))

        center = np.concatenate((np.zeros(n), np.zeros(m), sigma*nu*np.ones(n)))
        print(F(x, lamb, mu).size)
        solution = la.lu_solve(la.lu_factor(DF), -F(x, lamb, mu) + center)

        delta_x, delta_lamb, delta_mu = solution[:n], solution[n:n+m], solution[n+m:]

        return delta_x, delta_lamb, delta_mu

    def step_size(x, delta_x, mu, delta_mu):
        ''' Computes the step length for each iteration:
            Paramaters:
                x ((n, ) ndarray) current x
                d_x ((n, ) ndarray) change in current x
                mu ((n, ) ndarray) current lagrange mulgiplier
                d_mu ((n, ) ndarray) change incurrent lagrange mulgiplier
            Returns:
                alpha (float) step length for x
                delta (float) step length for mu
        '''
        mask_1 = delta_x < 0
        mask_2 = delta_mu < 0

        possible_deltas = -1/x[mask_1] * delta_x[mask_1]
        possible_alphas = -1/mu[mask_2] * delta_mu[mask_2]

        if possible_deltas.size == 0:
            delta = 1
        else:
            delta = np.min(possible_deltas)
        if possible_alphas.size == 0:
            alpha = 1
        else:
            alpha = np.min(possible_alphas)

        return delta, alpha

    #get intial point
    x, lam, mu = starting_point(A, b, c)
    i = 0
    nu = np.inner(x, mu) / n
    if abs(nu) < tol:
        return x, np.inner(c, x)
    while i < niter:
        #get directions
        direct_x, direct_lam, direct_mu = direction(x, lam, mu, nu, sigma=0.1)
        #get step sizes
        alph, delt = step_size(x, direct_x, mu, direct_mu)
        #get new x, lambda, and mu values
        x += delt * direct_x
        lam += alph * direct_lam
        mu += alph * direct_mu
        #update iteration count
        i += 1
        #get new duality measure
        nu = np.inner(x, mu) / n
        #check convergence
        if abs(nu) < tol:
            return x, np.inner(c, x)


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    raise NotImplementedError("Problem 5 Incomplete")


if __name__ == "__main__":

    j, k = 7, 5
    A, b, c, x = randomLP(j, k)
    point, value = interiorPoint(A, b, c)
    print(np.allclose(x, point[:k]))
