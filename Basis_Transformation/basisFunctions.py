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
        raise ValueError( "k is not in range for the bernstein polynomials." )
    if type( domain ) is not list or len( domain ) != 2:
        raise TypeError( "Domain is not a list or domain is the wrong size." )

    a, b = domain[ 0 ], domain[ -1 ]

    if b <= a:
        raise ValueError( "Domain is infeasible." )

    #bernstein function B_k^n(x)
    return lambda x: binom( n, k ) * ((( b - x ) / ( b - a ))**( n-k )) * ((( x - a ) / ( b- a ))**k)

def chebyshevFunc( n, domain=[-1, 1] ):
    '''Returns the chebyshev function of degree n
       Paramaters:
           n (int): the degree of the polynomial
           domain (list): the domain to be relevant on
       returns:
            T_n(x) (function): degree n chebyshev polynomial
    '''
    if type( domain ) is not list or len( domain ) != 2:
        raise TypeError( "Domain is not a list or domain is the wrong size." )

    a, b = domain[0], domain[-1]

    if b <= a:
        raise ValueError( "Domain is infeasible." )
    if n < 0:
        raise ValueError( "Invalid Chebyshev polynomial degree." )

    if n == 0:
        return lambda x : 1
    elif n == 1:
        return lambda x: 2 * x / (b-a) - (b+a) / (b-a)
    else:
        #recursion
        return lambda x: 2 * (2 * x / (b-a) - (b+a) / (b-a)) * chebyshevFunc(n-1, domain=domain)(x) - chebyshevFunc(n-2, domain=domain)(x)
