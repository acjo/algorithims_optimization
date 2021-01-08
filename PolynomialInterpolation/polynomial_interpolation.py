# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Caelan Osman
Math 323 Sec 2
Jan 7 2021
"""


import numpy as np
from matplotlib import pyplot as plt
import math


# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """

    #number of interpolation points
    n = xint.size
    #number of evaluation points
    m = points.size
    #denominators
    denoms = np.array([np.product(np.array([xint[j] - xint[k] for k in range(n) if j != k]))
                       for j in range(n)])

    def Lj(x, d, j, xvals):
        """evaluates the lagrange basis functions at the point x.
           Paramaters:
               x (float): value to evaluated
               d (float): denominator value
               xvals ((n, ) ndarray): x values to be interpolated
               j (int): the legrange basis function number
           returns:
               eval (float): evaluated value
        """
        s = xvals.size
        numer = np.product(np.array([x - xvals[k] for k in range(s) if k != j]))
        evaluation = numer / d
        return evaluation

    #lagrange matrix
    l_matrix = np.array([[Lj(x, denoms[j], j, xint) for x in points] for j in range(n)])
    #polynomial evaluation points
    p = np.array([sum(yint * l_matrix[:, j]) for j in range(m)])
    return p




# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.x = xint
        self.y = yint
        self.n = self.x.size
        self.w = np.ones(self.n)
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n - 1)
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]
            self.w[j] /= np.product(temp)

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        '''
        for x in self.x:
            mask = x == points
            points = points[~mask]
        '''
        p = np.array([sum(self.w * self.y / (x - self.x)) /
                      sum(self.w / (x - self.x)) for x in points])
        return p

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    raise NotImplementedError("Problem 7 Incomplete")


if __name__ == "__main__":
    #problem 1:
    '''
    n = 5
    runge = lambda x: 1 / ( 1 + 25 * x ** 2)
    x = np.linspace(-1, 1, n)
    y = runge(x)
    domain = np.linspace(-1, 1, 100)
    output = lagrange(x, y, domain)
    plt.plot(domain, runge(domain), 'c-', label='Original')
    plt.plot(domain, output, 'r-', label='Interpolation')
    plt.legend(loc='best')
    plt.show()
    '''
    #problem 3:
    '''
    n = 11
    runge = lambda x: 1 / ( 1 + 25 * x ** 2)
    x = np.linspace(-1, 1, n)
    y = runge(x)
    domain = np.linspace(-1, 1, 1000)
    domain = np.delete(domain, 0)
    domain = np.delete(domain, 998)
    b = Barycentric(x, y)
    output = b(domain)
    plt.plot(domain, runge(domain), 'c-', label='Original')
    plt.plot(domain, output, 'r-', label='Interpolation')
    plt.legend(loc='best')
    plt.show()
    '''


