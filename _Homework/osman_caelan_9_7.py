#osman_caelan_9_7.py

import numpy as np
from numpy.polynomial.legendre import leggauss as gauss

def gaussian_quadrature(f, n):
    ''' integrates f using gaussain quadrature of
        n+1 nodes
        Paramaters:
            f (lambda): function to integrate
            n (int): n+1 nodes to interpolate
    '''

    zeros, weights = gauss(n+1)

    approximation = np.sum(f(zeros) * weights)


    return approximation

def problem9_45():
    ''' Computes the gaussian quadrature estimate for
        |x| and cos(x) over [-1, 1]
    '''
    f1 = lambda x: np.abs(x)
    f2 = lambda x: np.cos(x)

    ns = np.arange(10, 110, 10)

    estimate_f1 = np.array([gaussian_quadrature(f1, n) for n in ns])
    estimate_f2 = np.array([gaussian_quadrature(f2, n) for n in ns])


    return estimate_f1, estimate_f2

