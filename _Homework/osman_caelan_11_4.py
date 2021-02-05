#osman_caelan_11_4.py

import time
import numpy as np
from autograd import grad
from numpy.linalg import norm
from matplotlib import pyplot as plt
from autograd import numpy as anp


def forward_difference(n, f, x, h=2*np.sqrt(np.finfo(float).eps)):
    '''Use forward difference quotient to approximate derivative
       Paramaters:
           n (int): the dimension of x
           f (lambda): callable function R^n to R
           x ((,n) ndarray): point to approximate derivative at
           h (float): the difference
       Returns:
           (float): the approximated derivative
    '''
    I = np.eye(n)
    J = np.array([(f(x + h * I[:, j]) -f(x)) / h for j in range(n)])

    return J


def compute_gradient(x):
    ''' return the gradient at point x of the function f: R^2 to R^2
        f(x, y) = (x / (x + 2+y**2) , y/ (x**2 + y**2))
        Paramters:
            x ((2, 2) ndarray): point to calculate gradient at (DF^T)
        Returns:
            ((2, 2) ndarray): Gradient matrix
    '''

    Df = lambda z: np.array([[(z[1]**2 + 2) / (z[0] + 2 + z[1]**2)**2, -2*z[0]*z[1] / (z[0] + 2 + z[1]**2)**2],
                             [-2*z[0]*z[1] / (z[0]**2 + z[1]**2)**2, (z[0]**2 - z[1]**2) / (z[0]**2 + z[1]**2)**2]])

    return Df(x).T




def optimal_h_forward_difference():
    x = np.array([2, 3])

    fx = lambda z: np.array([z[0]/ (z[0] + 2 + z[1]**2), z[1] / (z[0]**2 + z[1]**2)])

    ks = np.arange(2, 54)

    exact = compute_gradient(x)

    timing = []
    absolute_error = []
    for k in ks:
        start = time.time()
        estimate = forward_difference(2, fx, x, h= 1 / (2**k))
        end = time.time()
        timing.append(end - start)
        error = norm(exact[0] - estimate[0], ord=2) / norm(exact[0], ord=2)
        error += norm(exact[1] - estimate[1], ord=2) / norm(exact[1], ord=2)
        error /= 2
        absolute_error.append(1 - error)

    optimal = (ks[np.argmax(absolute_error)], timing[np.argmax(absolute_error)])
    fastest = (ks[np.argmin(timing)], min(timing))
    plt.plot(ks, absolute_error, 'c--', label='Error')
    plt.plot(ks, timing, 'm--', label = 'Time')
    plt.legend(loc='best')
    plt.title('Forward Difference\nBest h: 2^-' + str(optimal[0]) +'\nTime: '
              + str(optimal[1])+'\nFastest h: 2^-' + str(fastest[0]) +'\nFastest time: ' + str(fastest[1]))
    plt.show()

    return optimal



def centered_difference(n, f, x, h=1.4*(np.finfo(float).eps**(1/3))):
    '''Use Centered difference quotient to approximate derivative
       Paramaters:
           n (int): the dimension of x
           f (lambda): callable function R^n to R
           x ((,n) ndarray): point to approximate derivative at
           h (float): the difference
       Returns:
           (float): the approximated derivative
    '''
    I = np.eye(n)
    J = np.array([(f(x + h*I[:,j]) -f(x - h*I[:,j])) / (2*h) for j in range(n)])

    return J



def optimal_h_centered_difference():
    x = np.array([2, 3])

    fx = lambda z: np.array([z[0]/ (z[0] + 2 + z[1]**2), z[1] / (z[0]**2 + z[1]**2)])

    ks = np.arange(2, 54)

    exact = compute_gradient(x)

    timing = []
    absolute_error = []
    for k in ks:
        start = time.time()
        estimate = centered_difference(2, fx, x, h= 1 / (2**k))
        end = time.time()
        timing.append(end - start)
        error = norm(exact[0] - estimate[0], ord=2) / norm(exact[0], ord=2)
        error += norm(exact[1] - estimate[1], ord=2) / norm(exact[1], ord=2)
        error /= 2
        absolute_error.append(1 - error)

    optimal = (ks[np.argmax(absolute_error)], timing[np.argmax(absolute_error)])
    fastest = (ks[np.argmin(timing)], min(timing))
    plt.plot(ks, absolute_error, 'c--', label='Error')
    plt.plot(ks, timing, 'm--', label = 'Time')
    plt.legend(loc='best')
    plt.title('Centered Difference\nBest h: 2^-' + str(optimal[0]) +'\nTime: '
              + str(optimal[1])+'\nFastest h: 2^-' + str(fastest[0]) +'\nFastest time: ' + str(fastest[1]))
    plt.show()

    return optimal



def prob_18():

    def fdq1(f, x, h=1e-5):
        """Calculate the first order forward difference quotient of f at x."""
        return (f(x + h) -f(x)) / h

    def cdq2(f, x, h=1e-5):
        """Calculate the second order centered difference quotient of f at x."""
        return (f(x + h) - f(x - h)) / (2*h)

    def complex_step(f, x, h):
        """Uses complext_step differentiation to approximate f'(x)"""
        return np.imag(f(x + 1j*h) / h)

    x0 = 1.5
    fx = lambda x: (anp.sin(x)**3 + anp.cos(x)) / anp.exp(x)
    Symbolic = lambda x: (-2 * np.sin(x) + 3*np.sin(2*x)*np.sin(x) - 2*np.sin(x)**3 - 2*np.cos(x)) / (2*np.exp(x))
    exact = Symbolic(x0)
    ks = np.arange(1, 54)
    converge_forward, converge_centered, converge_complex = [], [], []
    accuracy_forward, accuracy_centered, accuracy_complex, accuracy_grad = [], [], [], []
    time_forward, time_centered, time_complex, time_grad = [], [], [], []
    for i, k in enumerate(ks):

        hval = 1/2**k
        if i != 0:
            converge_forward.append(abs(fdq1(fx, x0, h=hval) - exact) -
                                        abs(fdq1(fx, x0, h=1/2**ks[i-1]) - exact) / (ks[i] - ks[i-1]) )
            converge_centered.append(abs(cdq2(fx, x0, h=hval) - exact) -
                                         abs(cdq2(fx, x0, h=1/2**ks[i-1]) - exact) / (ks[i]- ks[i-1]) )
            converge_complex.append(abs(complex_step(fx, x0, h=hval) - exact) -
                                        abs(complex_step(fx, x0, h=1/2**ks[i-1]) - exact) / (ks[i] - ks[i-1]) )
        else:
            converge_forward.append(abs(fdq1(fx, x0, h=hval) - exact) -
                                        abs(fdq1(fx, x0, h=1/2**ks[i+1]) - exact) / (ks[i] - ks[i+1]) )
            converge_centered.append(abs(cdq2(fx, x0, h=hval) - exact) -
                                         abs(cdq2(fx, x0, h=1/2**ks[i+1]) - exact) / (ks[i]- ks[i+1]) )
            converge_complex.append(abs(complex_step(fx, x0, h=hval) - exact) -
                                        abs(complex_step(fx, x0, h=1/2**ks[i+1]) - exact) / (ks[i] - ks[i+1]) )

        start_for = time.time()
        accuracy_forward.append(abs(fdq1(fx, x0, h=hval) - exact))
        time_forward.append(time.time() - start_for)

        start_cen = time.time()
        accuracy_centered.append(abs(cdq2(fx, x0, h=hval) - exact))
        time_centered.append(time.time() - start_cen)

        start_com = time.time()
        accuracy_complex.append(abs(complex_step(fx, x0, h=hval) - exact))
        time_complex.append(time.time() - start_com)

        start_grad = time.time()
        accuracy_grad.append(abs(grad(fx)(x0) - exact))
        time_grad.append(time.time() - start_grad)


    fig, axs = plt.subplots(1, 3)
    ax = axs[0]
    ax.plot(ks, accuracy_forward, label='Forward Step')
    ax.plot(ks, accuracy_centered, label='Centered Step')
    ax.plot(ks, accuracy_complex, label='Complex Step')
    ax.plot(ks, accuracy_grad, label='Autograd')
    ax.set_title('Accuracy')
    ax.legend(loc='best')

    ax = axs[1]
    ax.plot(ks, time_forward, label='Forward Step')
    ax.plot(ks, time_centered, label='Centered Step')
    ax.plot(ks, time_complex, label='Complex Step')
    ax.plot(ks, time_grad, label='Autograd')
    ax.set_title('Computation Time')
    ax.legend(loc='best')

    ax = axs[2]
    ax.plot(ks, converge_forward, label='Forward Step')
    ax.plot(ks, converge_centered, label='Centered Step')
    ax.plot(ks, converge_complex, label='Complex Step')
    ax.set_title('Convergence Rates')
    ax.legend(loc='best')
    plt.suptitle('Numerical Differentiation Methods')
    plt.show()

    return

