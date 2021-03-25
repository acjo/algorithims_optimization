# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
Caelan Osman
Math 323 Sec. 2
March 23, 2021
"""

import cvxpy as cp
import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.
    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """

    #get necessary dimensions
    m, n = A.shape

    #define necessary matrices
    negI_mm = -np.eye(m)
    zero_nm = np.zeros((n, m))
    zero_mm = np.zeros((m, m))

    #Fromulate first two blocks of DF
    block_1 = np.column_stack((Q, zero_nm, -A.T))
    block_2 = np.column_stack((A, negI_mm, zero_mm))
    block_1_2 = np.vstack((block_1, block_2))

    #used for third block of DF
    zero_mn = np.zeros((m, n))

    def F(x, y, mu):
        '''Subroutine defining F
           Paramaters:
               x ((n, ) ndarray): variable in objective function
               y ((m, ) ndarray): slack variable in KKT
               mu ((m, ) ndarray): lagrange multipliers
           Returns:
               ((2m + n, ) ndarray): the objective function
        '''
        #first second and third elements defining F
        first = Q@x - A.T @ mu + c
        second = A@x - y - b
        third = np.diag(y) @ np.diag(mu) @ np.ones(m)
        return np.concatenate((first, second, third))

    def block3(y, mu):
        '''Subroutine grabbing the 3rd block matrix of DF
           Paramaters:
               y ((m, ) ndarray): slack variable in KKT
               mu ((m, ) ndarray): lagrange multipliers
           Returns
               block3 ((m, 2m + n) ndarray): the final block
        '''
        return np.column_stack((zero_mn, np.diag(mu), np.diag(y)))

    def direction(x, y, mu, nu, sigma=0.1):
        '''Subroutine getting the search directions
           Paramaters:
               x ((n, ) ndarray): variable in objective function
               y ((m, ) ndarray): slack variable in KKT
               mu ((m, ) ndarray): lagrange multipliers
               nu (float): duality measure
               sigma (float): scaling defaulting to 0.1
           Returns:
               delta_x ((n, ) ndarray): x search direction
               delta_y ((m, ) ndarray): y search direciton
               delta_mu((m, ) ndarray): mu search direction
        '''
        #grab DF and centering vector
        DF = np.vstack((block_1_2, block3(y, mu)))
        center = np.concatenate((np.zeros(n), np.zeros(m), sigma * nu * np.ones(m)))
        #solve system and return delta x, delta y, delta_mu as search directions
        solution = la.lu_solve(la.lu_factor(DF), -F(x, y, mu) + center)
        return solution[:n], solution[n:n+m], solution[n+m:]

    def step_size(mu, delta_mu, y, delta_y, tau=0.95):
        ''' Subroutine to grab the step size
            Paramaters:
                mu ((m, ) ndarray): current lagrange multiplier
                delta_mu ((m, ) ndarray): mu step direction
                y ((m, ) ndarray): current slack
                delta_y ((m, ) ndarray): slack step direction
                tau (float): backing off variable
            Returns:
                alpha (float): the step length
        '''
        mask_1 = delta_mu < 0
        mask_2 = delta_y < 0

        possible_betas = -mu[mask_1] / delta_mu[mask_1]
        possible_deltas = -y[mask_2] / delta_y[mask_2]

        #find beta
        if possible_betas.size == 0:
            beta = 1
        else:
            beta = tau * np.min(possible_betas)
        #find delta
        if possible_deltas.size == 0:
            delta = 1
        else:
            delta = tau * np.min(possible_deltas)

        #return alpha
        return min(beta, delta)


    #we now iterate to solve the convex optimization problem
    #get initial point
    x, y, mu = startingPoint(Q, c, A, b, guess)
    #initial duality
    nu = np.inner(y, mu) / m
    #iterate a max of niter times or until nu < tol
    for _ in range(niter):
        #calculate duality
        direction_x, direction_y, direction_mu = direction(x, y, mu, nu)
        #get step size
        alpha = step_size(mu, direction_mu, y, direction_y)
        #get new x, y, mu vectors
        x += alpha * direction_x
        y += alpha * direction_y
        mu += alpha * direction_mu
        #check convergence
        nu = np.inner(y, mu) / m
        if abs(nu) < tol:
            break

    return x,  0.5 * np.inner(x, Q @ x) + np.inner(c, x)


def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    #create tent pole configuration
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()

    #set initial guesses
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    #grab Dirichlet energy
    H = laplacian(n)
    #set c
    c =  -1 / ((n-1)**2) * np.ones(x.size)
    #set A
    A = np.eye(x.size)

    #calculate solution
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))
    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.title('Circus Tent')
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    R = 1.13

    with open(filename) as infile:
        content = infile.readlines()

    #year and stock vectors
    years = np.array([line.strip().split(' ')[0] for line in content]).astype(np.float64)
    stock = np.array([line.strip().split(' ')[1:] for line in content]).astype(np.float64)

    #get return rates mu_i
    rates = np.array([np.mean(row) for row in stock])

    #get the covariance matrix
    Q = np.cov(stock)

    #calculate the optimal portfolio with short selling
    x = cp.Variable(Q.shape[1])
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q))
    constraints = [sum(x) == 1, x @ rates == R]
    prob = cp.Problem(objective, constraints)
    solution = prob.solve()
    with_ss = x.value

    #calculate the optimal portfolio without short selling
    x = cp.Variable(Q.shape[1], nonneg=True)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q))
    constraints = [sum(x) == 1, x @ rates == R]
    prob = cp.Problem(objective, constraints)
    solution = prob.solve()
    with_oss = x.value

    return with_ss, with_oss



if __name__ == "__main__":
    #test problem 1 and 2
    '''
    Q = np.array([[1, -1],
                  [-1, 2]])

    c = np.array([-2, -6])

    A = np.array([[-1, -1],
                  [1, -2],
                  [-2, -1],
                  [1, 0],
                  [0, 1]])

    b = np.array([-2, -2, -3, 0, 0])

    x0 = np.array([0.5, 0.5])
    y0 = np.ones(5)
    mu0 = np.ones(5)
    guess =(x0, y0, mu0)

    solution =  qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False)
    actual = np.array([2/3., 4/3.])
    value = -8.222222

    print(np.allclose(solution[0], actual))
    print(np.allclose(solution[1], value))

    #problm 3
    circus(n=15)

    print(portfolio()[0])
    print()
    print(portfolio()[1])
    '''
