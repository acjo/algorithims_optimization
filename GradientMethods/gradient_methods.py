# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Caelan Osman
Math 323 Sec. 2
Feb 22, 2021
"""

import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize_scalar as ms

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    #set loop coutner
    i = 0
    converge = False
    while i < maxiter:
        #set function
        phi = lambda alpha: f(x0 - alpha* Df(x0).T)
        #comptue optimal alpha with 1d optimization
        #compute 1d optimization
        alpha = ms(phi).x
        #get next iteration
        x0 = x0 - alpha*Df(x0).T
        #iterate loop counter
        i += 1
        #check convergence
        if la.norm(Df(x0), ord=np.inf) < tol:
            converge = True
            break

    #return approx min, convergence, number of iterations
    return x0, converge, i


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #set intial r values copy because arrays are immutable
    r0 = Q@x0 - b
    r1 = np.copy(r0)
    #set intial conjugate gradient direction
    d0 = -np.copy(r0)
    #loop counter, mixiter, and convergence bool
    i, maxiter = 0, x0.size
    converge = False

    while i < maxiter:
        #iterate alpha
        alpha = np.inner(r0, r0) / np.inner(d0, Q @ d0)
        #iterate x
        x0 = x0 + alpha * d0
        #iterate r
        r1 = r0 + alpha * Q @ d0
        #iterate beta
        beta = np.inner(r1, r1) / np.inner(r0, r0)
        #iterate d
        d0 = - r1 + beta * d0
        #reassign r0
        r0 = r1
        #iterate loop counter
        i += 1
        #check convergence
        if la.norm(r0) < tol:
            converge = True
            break

    return x0, converge, i



# Problem 3
def nonlinear_conjugate_gradient(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #get r0, d0
    r0 = -Df(x0).T
    d0 = np.copy(r0)
    #set current line search function
    phi = lambda alpha: f(x0 + alpha*d0)
    #optimize alpha
    alpha = ms(phi).x
    #get next iteration
    x0 = x0 + alpha*d0

    #intialize loop counter
    i = 1
    while la.nrom(r0, ord=2) and i < maxiter:
        #get next r value
        r1 = -Df(x0).T
        #iterate beta
        beta = np.inner(r1, r1) / np.inner(r0, r0)
        #iterate conjugate direction
        d0 = r1 + beta*d0
        #set current line search function
        phi = lambda alpha: f(x0 + alpha*d0)
        #optimize alpha
        alpha = ms(phi).x
        x0 = x0 + alpha * d0
        r0 = r1
        i += 1


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":

    #problem 1
    #test 1
    '''
    #f(x,y,z) =x^4 + y^4 + z^4
    fx = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df = lambda x: np.array([4*x[0]**3, 3*x[1]**3, 4*x[2]**3])
    x0 = np.array([2, -3, 6])
    optim = steepest_descent(fx, Df, x0, tol=1e-20, maxiter=20)
    print(optim)
    '''
    #test 2 (rosenbrock)
    #f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    '''
    fx  = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    Df = lambda x: np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2), 200*(x[1] - x[0]**2)])
    x0 = np.array([-2, 3])
    optim = steepest_descent(fx, Df, x0, tol=1e-10, maxiter=10000000)
    print(optim)
    '''

    #problem 2
    '''
    #test 1
    Q = np.array([[2, 0], [0, 4]])
    b = np.array([1, 8])
    x0 = np.array([13, -4])
    print(conjugate_gradient(Q, b, x0))
    #test prob 1
    fx = lambda x: x[0]**2 + 2*x[1]**2 - x[0] - 8*x[1]
    Df = lambda x: np.array([2*x[0] - 1, 4*x[1] - 8])
    x0 = np.array([3, 4])
    print(steepest_descent(fx, Df, x0))
    #test 2
    n=4
    A = np.random.random((n, n))
    Q = A.T @ A
    b, x0 = np.random.random((2, n))
    print(conjugate_gradient(Q, b, x0))
    '''

