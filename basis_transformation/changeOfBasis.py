#change_of_basis.py

import numpy as np
from numpy.fft import fft
from scipy.special import binom
from matplotlib import pyplot as plt
import basisFunctions as bf


def cKronecker( delta ):
    """Conjugate Kronecker delta function
    """
    if delta == 0:
        return 0
    else:
        return 1

def transitionMatrixBTC( n ):
    """Creates the transition matrix from Bernstein to Chebyshev polynomials
       n is the polynomial degree, the matrix will be ((n+1), (n+1))
    """
    C = np.zeros(( n+1, n+1 ))
    for j in range( n+1 ):
        for k in range( n+1 ):
            C[ j, k ] = ( ( cKronecker( j ) + 1 ) * binom( n, k ) ) / 4**( n+j )
            C[ j, k ] *= np.sum( [ (-1)**(j-i) * binom(2*j, 2*i) * binom( 2*( k+i ), k+i ) * binom( 2 * ( n+j-k-i ), n+j-k-i) / binom( n+j, k+i ) for i in range( j + 1 ) ])
    return C

def transitionMatrixCTB( n ):
    """Creates the transition matrix from Chebyshev to Bernstein polynomials
       n is the polynomial degree, the matrix will be ((n+1), (n+1))
    """
    C = np.zeros(( n+1, n+1 ))
    for j in range( n+1 ):
        for k in range( n+1 ):
            C[ j, k ] = 1 / binom( n, j )
            upper = min( j, k )
            lower = max( 0, j+k-n )
            C[ j, k ] *= np.sum( [ ( -1 )**( k-i ) * binom( 2*k, 2*i ) * binom( n-k, j-i ) for i in range( lower, upper + 1) ] )

    return C

def bernsteinToChebyshev( b, domain = [0, 1], matrix=False ):
    """Changes from the bernstein to chebyshev basis
       Paramaters:
           b ((n+1, ) ndarray): coordinates in the degree-n Bernstein basis
           matrix (bool): boolean to use fft or matrix transform
       Returns:
           coeffs ((n+1, ) ndarray): coordinates in the degree-n Chebyshev basis
    """
    n = b.size - 1
    if not matrix:

        bBasis = [ bf.bernsteinFunc( n, i, domain=domain ) for i in range( 0, n+1 ) ]

        polyB = lambda x: np.sum( [ b[ i ] * f( x ) for i, f in enumerate( bBasis ) ] )

        extremizers = np.cos( ( np.pi * np.arange( 2*n ) ) / n )
        samples = np.array([ polyB( extreme ) for extreme in extremizers ])

        coeffs = np.real(fft(samples))[:n+1] / n
        coeffs[0] /= 2
        coeffs[-1] /= 2

        return coeffs

    else:
        #define conjugate Kronecker delta function
        def cKronecker( delta ):
            if delta == 0:
                return 0
            else:
                return 1
        C = transitionMatrixBTC( n )

        return C @ b

if __name__ == "__main__":

    print(transitionMatrixBTC(1))
    print()
    print(transitionMatrixCTB( 1 ))







