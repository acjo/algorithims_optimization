#bernstein_functions.py
import numpy as np
from scipy.special import binom

def bernsteinFunc( n, k, domain=[0, 1] ):
    ''' returns the bernstein function for a given non-negative integer
        n and an integer 0 <= k <=  n
        Paramaters:
            n ( int ): degree of bernstein polynomials
            k ( int ): ranging integer
            domain (list): the desired domain for the polynomials to be valid over
        Returns:
            func ( function ): the bernstein polynomial
        Raises:
            ValueError: if k < 0 or k > n
    '''
    if k < 0 or k > n:
        raise ValueError('k is not in range for the bernstein polynomials')
    a = domain[0]
    b = domain[1]

    #bernstein function B_k^n(x)
    return lambda x: binom( n, k ) * (( x - a ) / ( b-a ))**k*(( x - b ) / ( b - a )) ** ( n - k )

def chebyshevFunc( n, domain=[-1, 1] ):
    '''Returns the chebyshev function of degree n
       Paramaters:
           n (int): the degree of the polynomial
           domain (list): the domain to be relevant on
       returns:
            T_n(x) (function): degree n chebyshev polynomial
    '''
    a, b = domain[0], domain[-1]


    if n == 0:
        return lambda x : 1
    elif n == 1:
        return lambda x: 2 * x / (b-a) - (b+a) / (b-a)
    else:
        #recursion
        return lambda x: 2 * (2 * x / (b-a) - (b+a) / (b-a)) * chebyshevFunc(n-1, domain=domain)(x) - chebyshevFunc(n-2, domain=domain)(x)
