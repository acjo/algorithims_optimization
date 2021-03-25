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
    zero_nn = np.zeros((n, n)
    )
    #m x n zero matrix
    zero_mn = np.zeros((m, n))
    #m x m zero matrix
    zero_mm = np.zeros((m, m))
    #n x n identity amtrix
    I_nn = np.eye(n)
    #define first two block matrices of DF
    block_1 = np.column_stack((zero_nn, A.T, I_nn))
    block_2 = np.column_stack((A, zero_mm, zero_mn))
    #stack together
    block_1_2 = np.vstack((block_1, block_2))

    #used for third block matrix
    zero_nm = np.zeros((n, m))

    def F(x, lamb, mu):
        ''' defines the vector vaued function used the interior point problem
            Paramaters:
                x ((n,) ndarray): variable in objective function
                lambda ((m, ) ndarray): KKT condition lagrange multiplier
                mu ((n, ) ndarray): KKT condition lagrange multiplier
            Returns:
                ((2n +m, ) ndaraay): output of function
        '''
        first = A.T @ lamb + mu - c
        second = A @ x - b
        third  = np.diag(mu) @ x
        return np.concatenate((first, second, third))

    def block3(x, mu):
        '''Returns the last block row in the block matrix defining the
           Ferchet derivative.
            Paramaters:
                x ((n,) ndarray): variable in objective function
                mu ((n, ) ndarray): KKT condition lagrange multiplier
            Returns:
                ((n,n2n+m) ndaraay): output of function

        '''
        M = np.diag(mu)
        X = np.diag(x)
        #return bottom block
        return np.column_stack((M, zero_nm, X))

    def direction(x, lamb, mu, nu, sigma=0.1):
        '''searching direction subroutine
           Paramaters:
               x ((n, ) ndarray): variable in objective function
               lamb ((m, ) ndarray): slack variable in KKT
               mu ((n, ) ndarray): lagrange multipliers
               nu (float): duality measure
               sigma (float): defailting to 0.1 used for scaling
            Returns:
                delta_x ((n, ) ndarray): delta x
                delta_lambda ((m, ) ndarray): delta lambda
                delta_mu ((n, ) ndarray): delta mu
        '''
        #get DF matrix
        DF = np.vstack((block_1_2, block3(x, mu)))
        #get centering vector
        center = np.concatenate((np.zeros(n), np.zeros(m), sigma*nu*np.ones(n)))

        #solve the system and return delta_x, delta_lambda, delta_mu as the directions
        solution = la.lu_solve(la.lu_factor(DF), -F(x, lamb, mu) + center)
        return solution[:n], solution[n:n+m], solution[n+m:]


    def step_size(x, delta_x, mu, delta_mu):
        ''' Computes the step length for each iteration:
            Paramaters:
                x ((n, ) ndarray): current x
                delta_x ((n, ) ndarray): change in current x
                mu ((n, ) ndarray): current lagrange mulgiplier
                delta_mu ((n, ) ndarray): change incurrent lagrange mulgiplier
            Returns:
                alpha (float): step length for x
                delta (float): step length for mu
        '''
        #get mask on delta_x and delta_mu
        mask_1 = delta_x < 0
        mask_2 = delta_mu < 0

        #get arrays of possible alphas and deltas
        possible_alphas = -mu[mask_2] / delta_mu[mask_2]
        possible_deltas = -x[mask_1] / delta_x[mask_1]

        #find alpha
        if possible_alphas.size == 0:
            alpha = 1
        else:
            alpha = 0.95 * np.min(possible_alphas)
        #find delta
        if possible_deltas.size == 0:
            delta = 1
        else:
            delta = 0.95 * np.min(possible_deltas)


        return alpha, delta

    #get intial point
    x, lamb, mu = starting_point(A, b, c)
    nu = np.inner(x, mu) / n
    #iterate at most niter times
    for _ in range(niter):
        #get directions
        direction_x, direction_lamb, direction_mu = direction(x, lamb, mu, nu, sigma=0.1)
        #get step sizes
        alpha, delta = step_size(x, direction_x, mu, direction_mu)
        #get new x, lambda, and mu values
        x += delta * direction_x
        lamb += alpha * direction_lamb
        mu += alpha * direction_mu
        #get new duality measure
        nu = np.inner(x, mu) / n
        #check convergence
        if abs(nu) < tol:
            break

    return x, np.inner(c, x)


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    #convert text file to numpy array of ys and xs
    with open(filename) as infile:
        data = infile.readlines()
    data = [line.strip().split(' ') for line in data]
    data = np.array(data).astype(np.float64)

    #initialize c and y vector
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    #set up A matrix
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    #get interior point solution
    sol = interiorPoint(A, y, c, niter=15)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    #use linear regression
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0, 10, 10)
    plt.plot(domain, domain*slope + intercept, 'm--', label='Linear Regression')
    plt.plot(data[:, 1], data[:, 0], 'bx', label='Data')
    plt.plot(data[:, 1], data[:, 1] * beta + b, 'c-.', label='Least Absolute Deviation')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":

    #test for problem 1 through 4
    '''
    j, k = 7, 5
    A1, b1, c1, x1 = randomLP(j, k)
    point, value = interiorPoint(A1, b1, c1)
    print(np.allclose(x1, point[:k]))

    j, k = 36, 24
    A1, b1, c1, x1 = randomLP(j, k)
    point, value = interiorPoint(A1, b1, c1)
    print(np.allclose(x1, point[:k]))
    '''

    #test for problem 5
    #leastAbsoluteDeviations()
