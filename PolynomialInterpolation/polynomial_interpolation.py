# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Caelan Osman
Math 323 Sec 2
Jan 7 2021
"""


import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy.interpolate import BarycentricInterpolator
from numpy.fft import fft


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
        #used for checking evaluation points
        self.key_vals = {x : yint[j] for j, x in enumerate(xint)}
        #saving x, y interpolating points
        self.x = xint
        self.y = yint
        #size of both x and y
        self.n = self.x.size
        self.w = np.ones(self.n)
        self.C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n - 1)
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint, j)) / self.C
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
        #initialzing output values
        vals = np.zeros(points.size)
        #iterate through points
        for i, point in enumerate(points):
            #if current point is an interpolating x point set current value to the corresponding
            #interpolating y point
            if point in self.key_vals:
                vals[i] = self.key_vals[point]
            #otherwise evaluate as normal
            else:
                vals[i] = sum(self.w * self.y / (point - self.x)) / sum(self.w / (point - self.x))

        return vals

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #new_C and what the factor is for factor * self.C = new_C
        new_C = (max(np.max(xint), np.max(self.x)) - min(np.min(xint), np.min(self.x))) / 4
        factor = new_C / self.C

        def update_weights(new_x):
            ''' Updates weights
                Paramaters:
                    new_x ((1, )ndarray): new x value to update the weights with
            '''
            self.w = np.array([w / (np.prod((self.x[i]- new_x) / self.C)) for i, w in enumerate(self.w)])

        def new_weights(new_x):
            '''Calculates the new weight
               Paramaters:
                   new_x (float): the value to calculate the new weight for
            '''
            update_weights(np.array([new_x]))
            weight = 1 / np.prod((new_x - self.x) / self.C)
            self.x = np.concatenate((self.x, [new_x]))
            self.w = np.concatenate((self.w, [weight]))

        #update and calculate new weights
        for xval in xint:
            new_weights(xval)


        #increment self.n
        self.n += xint.size
        #update weights by multiplying everything by the factor
        self.w *= factor**(self.n-1)
        #update and store the new_C
        self.C = new_C
        #concatenate the new yvals to self.y
        self.y = np.concatenate((self.y, yint))
        #update key value pairs
        self.key_vals = {x : self.y[i] for i, x in enumerate(self.x)}


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #all ns
    ns = [2**i for i in range(2, 9)]
    #the domain for plotting
    domain = np.linspace(-1, 1, 500)
    #function we are interpolating
    runge = lambda x: 1 / (1 + 25 * x**2)
    #error of interpolating polynomials for equally spaced points
    error_poly = []
    #error of interpolating polynomials for chebyshev extremizers
    error_cheby = []
    for n in ns:
        #equal and chebyshev extremizers
        equal = np.linspace(-1, 1, n)
        cheby_pts = np.array([np.cos(j*np.pi / n) for j in range(0, n+1)])
        #normal and chebyshev interpolation
        poly = BarycentricInterpolator(equal)
        cheby_poly = BarycentricInterpolator(cheby_pts)
        #set y points
        poly.set_yi(runge(equal))
        cheby_poly.set_yi(runge(cheby_pts))
        #calculate error
        error_poly.append(la.norm(runge(domain) - poly(domain), ord=np.inf))
        error_cheby.append(la.norm(runge(domain) - cheby_poly(domain), ord=np.inf))

    #plot
    plt.loglog(ns, error_cheby, 'm--', label='Interpolation at Chebyshev Extremes')
    plt.loglog(ns, error_poly, 'c:',label='Interpolation at equally spaced points')
    plt.legend(loc='best')
    plt.title('Interpolation Error')
    plt.show()


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
    #using Algorithm 9.1 in the book
    #get chebyshev points
    y = np.cos((np.pi * np.arange(2*n)) / n)
    #function values at chebyshev points
    samples = f(y)
    #compute fft and divide by n
    coeffs = np.real(fft(samples))[:n+1] / n
    #divide first and last by 2
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs

# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load('airdata.npy')
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    ax1 = plt.subplot(121)
    ax1.plot(domain, poly(domain), 'r-', label= 'n = ' + str(n))
    ax1.legend(loc='best')
    ax2 = plt.subplot(122)
    ax2.plot(domain, data, 'b', label='Data')
    ax2.legend(loc='best')
    plt.suptitle("Airquality")
    plt.show()


if __name__ == "__main__": #problem 1:
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
    b = Barycentric(x, y)
    output = b(domain)
    plt.plot(domain, runge(domain), 'c-', label='Original')
    plt.plot(domain, output, 'r-', label='Interpolation')
    plt.legend(loc='best')
    plt.show()
    '''
    #problem 4:
    '''
    n = 11
    runge = lambda x: 1 / (1 + 25 * x**2)
    xvals_original = np.linspace(-1, 1, n)
    xvals_1 = xvals_original[1::2]
    xvals_2 = xvals_original[::2]
    domain = np.linspace(-1, 1, 1000)
    bary = Barycentric(xvals_1, runge(xvals_1))

    bary_2 = Barycentric(xvals_original, runge(xvals_original))
    plt.plot(domain, bary_2(domain),linewidth=6, label='Not added')
    plt.plot(domain, runge(domain), label='Original')
    plt.plot(domain, bary(domain), label='Odd Points, n = ' + str(n))
    bary.add_weights(xvals_2, runge(xvals_2))
    #bary.add_weights(xvals_2[0], runge(xvals_2[0]))
    bary.add_weights(xvals_2[1], runge(xvals_2[1]))
    bary.add_weights(xvals_2[2], runge(xvals_2[2]))
    bary.add_weights(xvals_2[3], runge(xvals_2[3]))
    bary.add_weights(xvals_2[4], runge(xvals_2[4]))
    bary.add_weights(xvals_2[5], runge(xvals_2[5]))
    plt.plot(domain, bary(domain),'k', label='All points, n = ' + str(n))
    plt.legend(loc='best')
    plt.show()
    '''
    #problem 5:
    #prob5()

    #problem 6:
    '''
    f = lambda x: -3 + 2*x**2 - x**3 + x**4
    pcoeffs = [-3, 0, 2, -1, 1]
    ccoeffs = np.polynomial.chebyshev.poly2cheb(pcoeffs)
    myccoeffs = chebyshev_coeffs(f, 4)
    print(np.allclose(ccoeffs, myccoeffs))
    '''

    #problem 7:
    #prob7(200)


