# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Caelan Osman
Math 323 Sec. 2
January 27, 2020
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.stats import norm


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        #chech that polytype is an acceptable type
        if polytype != 'legendre' and polytype != 'chebyshev':
            raise ValueError('The polynomial type needs to be either legendre or chebyshev')

        #set poly and n attributes
        self.poly = polytype
        self.n = n

        #set the reciprocal weight function
        if self.poly == 'legendre':
            self.w = lambda x: 1
        else:
            self.w = lambda x: np.sqrt(1 - x**2)

        #get quadrature points and weights
        self.points, self.weights, = self.points_weights(self.n)


    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #helper functions for beta and the measure
        def beta(k):
            if self.poly == 'legendre':
                b = k**2 / (4*k**2 - 1)
                return k**2 / (4*k**2 - 1)
            else:
                if k == 1:
                    return 1/2.
                else:
                    return 1/4.
        def measure():
            if self.poly == 'legendre':
                return 2
            else:
                return np.pi

        #initialize Jacobi matrix
        J = np.zeros((n, n))
        #set values for the corresponding type
        for i in range(n):
            if i == 0:
                J[i, i+1] = np.sqrt(beta(i+1))
            elif i == n-1:
                J[i, i-1] = np.sqrt(beta(i))
            else:
                J[i, i+1] = np.sqrt(beta(i+1))
                J[i, i-1] = np.sqrt(beta(i))

        #get quadrature points and eigenvectors
        points, vecs = la.eigh(J)
        #calculate the weights
        weights = np.array([measure()*vecs[0, i]**2 for i in range(n)])


        return np.real(points), np.real(weights)


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        approx = 0
        for i, point in enumerate(self.points):
            approx += (f(point) * self.w(point) * self.weights[i])

        return approx

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x: f( x * (b-a) / 2 + (a + b) / 2)
        return (b - a) * self.basic(h) / 2

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x, y: f( x*(b1 - a1) / 2 + (a1 + b1) / 2, y*(b2 - a2) / 2 + (a2 + b2) / 2)
        approx = 0
        for i, pointy in enumerate(self.points):
            for j, pointx in enumerate(self.points):
                approx += (self.weights[i] * self.weights[j] * h(pointx, pointy) * self.w(pointx) * self.w(pointy))

        return (b1 - a1) * (b2 - a2) * approx / 4


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    #standard normal pdf
    fx = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)
    #get the exact value of the integration
    exact = norm.cdf(2) - norm.cdf(-3)
    ns = np.arange(5, 55, 5)
    #get error for legendre and chevyshev gaussian quadrature
    error_legen = np.array([abs(exact - GaussianQuadrature(n).integrate(fx, -3, 2)) for n in ns])
    error_cheb = np.array([abs(exact - GaussianQuadrature(n, 'chebyshev').integrate(fx, -3, 2)) for n in ns])
    error_scipy = np.array([abs(exact - quad(fx, -2, 3)[0]) for _ in ns])

    plt.semilogy(ns, error_legen, 'm-.', label='Error Legendre')
    plt.semilogy(ns, error_cheb, 'b--', label='Error Chebyshev')
    plt.semilogy(ns, error_scipy, 'r-', label='Error Scipy.Quad')
    plt.legend(loc='best')
    plt.title('Quadrature Error')
    plt.show()



if __name__ == '__main__':
    #problem 1 & 2
    '''
    test_points = np.array([(-1/3) * np.sqrt(5 + 2*np.sqrt(10/7)),
                             (-1/3) * np.sqrt(5 - 2*np.sqrt(10/7)),
                            0,
                             (1/3) * np.sqrt(5 - 2*np.sqrt(10/7)),
                             (1/3) * np.sqrt(5 + 2*np.sqrt(10/7))])

    test_weights = np.array([(322 - 13*np.sqrt(70)) / 900,
                             (322 + 13*np.sqrt(70)) / 900,
                             128 / 225,
                             (322 + 13*np.sqrt(70)) / 900,
                             (322 - 13*np.sqrt(70)) / 900])
    quad = GaussianQuadrature(5, 'legendre')

    print(np.allclose(quad.points, test_points))
    print(np.allclose(quad.weights, test_weights))
    '''

    #problem 3:
    '''
    fx = lambda x: 1 / np.sqrt(1 - x**2)
    n = 200

    Gauss = GaussianQuadrature(n, 'chebyshev')
    integral = Gauss.basic(fx)
    print(integral)
    print(quad(fx, -1, 1, )[0])
    '''

    #problem 4:
    '''
    fx = lambda x: np.cos(x)
    n = 30
    Gauss = GaussianQuadrature(n, 'chebyshev')
    integral = Gauss.integrate(fx, 0, 4)
    print(integral)
    '''

    #problem 5:
    #prob5()

    #problem 6:
    '''
    fx = lambda x, y: np.sin(x) + np.cos(y)
    n = 100
    Gauss = GaussianQuadrature(n, 'legendre')
    integral = Gauss.integrate2d(fx, -10, 10, -1, 1)
    scipy_integral = nquad(fx, [[-10, 10], [-1, 1]])[0]
    print(np.allclose(integral, scipy_integral))
    '''
