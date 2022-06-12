# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Caelan Osman
June 11, 2022
"""

from logging import makeLogRecord
import sys
from turtle import back
import numpy as np
from scipy.optimize import golden
from scipy.optimize import newton
from scipy.optimize import newton_krylov
from scipy.optimize.linesearch import scalar_search_armijo
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import grad

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=25):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    converged = False
    # intial guess
    x0 = (a + b) / 2
    # golden ratio
    phi = (1 + np.sqrt(5)) / 2

    for i in range(maxiter):

        # shrinking factor on interval
        c = (b - a) / phi

        # define new end/start points
        a_tilde = b - c
        b_tilde = a + c

        # interval check containing the minimizer 
        if f(a_tilde) < f(b_tilde):
            b = b_tilde
        else:
            a = a_tilde

        # next guess
        x1 = (a + b) / 2

        # check convergence
        if np.abs(x0 - x1) < tol:
            converged = True
            break

        x0 = x1

    return x1, converged , i + 1

# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=25):
    """Use Newton"s method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    converged = False
    for i in range(maxiter):

        # calculate new point
        x1 = x0 - (df(x0) / d2f(x0))

        # check convergence
        if np.abs(x1 - x0) < tol:
            converged = True
            break
        # reassign for next iteration
        x0 = x1

    return x1, converged, i + 1

# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=25):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    converged = False
    for i in range(maxiter):
        
        # calculate new guess
        x2 = (x0*df(x1) - x1*df(x0)) / (df(x1) - df(x0))

        # check convergence
        if np.abs(x2 - x1) < tol:
            converged = True
            break

        # reassign for next loop
        x0 = x1
        x1 = x2


    return x2, converged, i + 1

# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """

    Dfp = np.inner(Df(x), p)
    fx = f(x)

    while (f(x + alpha*p) > fx + c*alpha*Dfp):
        alpha *= rho
    return alpha


def main(key):

    if key == "1":
        # calculate minimzer
        f = lambda x: np.exp(x) - 4*x
        a = 0
        b = 3
        my_opt = golden_section(f, a, b,tol=1e-8, maxiter=5000)
        # get scipy"s answer
        scipy_opt = golden(f, brack=(0, 3), tol=1e-8)

        # plot function and minimizer
        domain = np.arange(a, 3, 1e-4)
        fig = plt.figure()
        fig.set_dpi(150)
        ax = fig.add_subplot(111)
        ax.plot(domain, f(domain), "r-", label=r"$f(x) = e^x - 4x$")
        ax.plot(my_opt[0], f(my_opt[0]), "bo", markersize=4, label="calculated minimizer: {}".format(round(my_opt[0], 3)))
        ax.plot(scipy_opt, f(scipy_opt), "ko", markersize=2, label="scipy's minimizer: {}".format(round(scipy_opt, 3)))
        ax.legend(loc="best")
        plt.show()
        
        assert np.allclose(my_opt[0], scipy_opt)

    elif key == "2":
        # set function and derivatives and initial values
        f = lambda x: x**2 + np.sin(5*x)
        df = lambda x: 2*x + 5*np.cos(5*x)
        d2f = lambda x: 2 - 25*np.sin(5*x)
        x0 = 0

        # call my function
        my_root, converged, iters = newton1d(df, d2f, x0, tol=1e-10, maxiter=5000)

        # call scipy"s function
        root = newton(df, x0=0, fprime=d2f, tol=1e-10, maxiter=500)

        # plot
        domain = np.arange(-2, 2, 1e-4)
        fig = plt.figure()
        fig.set_dpi(150)
        ax = fig.add_subplot(111)
        ax.plot(domain, f(domain), "r-", label=r"$f(x) = x^2 + \sin(5x)$")
        ax.plot(my_root, f(my_root), "bo", markersize=4, label="calculated minimizer: {}".format(round(my_root, 3)))
        ax.plot(root, f(root), "ko", markersize=2, label="scipy's minimizer: {}".format(round(root, 3)))
        ax.legend(loc="best")
        plt.show()

        assert np.allclose(my_root, root)

        
    elif key == "3":
        f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
        df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
        x0 = 0
        x1 = -1

        my_root, converged, iters = secant1d(df, x0, x1, tol=1e-10, maxiter=5000)
        root = newton(df, x0=x0, tol=1e-10, maxiter=5000)

        domain = np.arange(-3, 3, 1e-4)
        fig = plt.figure()
        fig.set_dpi(150)
        ax = fig.add_subplot(111)
        ax.plot(domain, f(domain), "r-", label=r"$f(x) = 2x + \sin(x) + \sin(10x)$")
        ax.plot(my_root, f(my_root), "bo", markersize=4, label="calculated minimizer: {}".format(round(my_root, 3)))
        ax.plot(root, f(root), "ko", markersize=4, label="scipy's minimizer: {}".format(round(root, 3)))
        ax.legend(loc="best")
        plt.show()

    elif key == "4":
        f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
        Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])

        x = anp.array([150., 0.03, 40.])
        p = anp.array([-0.5, -100., -4.5])
        phi = lambda alpha: f(x + alpha*p)
        dphi = grad(phi) 
        alpha, _ = scalar_search_armijo(phi, phi(0.), dphi(0.))
        print("scipy's alpha:", alpha)

        alpha = backtracking(f, Df, x, p)
        print("my alpha:", alpha)
        print('They should be similar')

    elif key == "all":
        main("1")
        main("2")
        main("3")
        main("4")
    else:
        raise ValueError ("{} is an incorrect problem specification.".format(key))

    return

if __name__ == "__main__":

    if len(sys.argv) == 2:
        main(sys.argv[1])