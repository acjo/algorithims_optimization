#change_of_basis.py

import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from scipy.special import binom
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
    if type( n ) is not int:
        raise TypeError( "n needs to be an integer." )

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
    if type( n ) is not int:
        raise TypeError( "n needs to be an integer." )

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

    if type( b ) is not np.ndarray:
        raise TypeError( "Bernsetin coefficeints is not an array." )
    if type( domain ) is not list or len( domain ) != 2:
        raise TypeError( "Domain is not a list or domain is the wrong size." )
    if domain[ -1 ] <= domain[ 0 ]:
        raise ValueError( "Domain is infeasible." )
    if type( matrix ) is not bool:
        raise TypeError( "'Matrix' needs to be a boolean value." )


    n = b.size - 1

    if matrix:
        T = transitionMatrixBTC( n )
        return T @ b

    else:

        bBasis = [ bf.bernsteinFunc( n, i, domain=domain ) for i in range( 0, n+1 ) ]

        polyB = lambda x: np.sum( [ b[ i ] * f( x ) for i, f in enumerate( bBasis ) ] )

        extremizers = np.cos( ( np.pi * np.arange( 2*n ) ) / n )
        samples = np.array([ polyB( extreme ) for extreme in extremizers ])

        coeffs = np.real( fft( samples ) )[ :n+1 ] / n
        coeffs[ 0 ] /= 2
        coeffs[ -1 ] /= 2

        return coeffs

def chebyshevToBernstein( c, domain=[0, 1], matrix=True ):

    if type( c ) i snot np.ndarray:
        raise TypeError( "Chebysehv coffeicients is not an array." )
    if type( domain ) is not list or len( domain ) != 2:
        raise TypeError( "Domain is not a list or domain is the wrong size." )
    if domain[ -1 ] <= domain[ 0 ]:
        raise ValueError( "Domain is infeasible." )
    if type( matrix ) is not bool:
        raise TypeError( "'Matrix' needs to be a boolean value." )


    n = c.size - 1

    if matrix:
        T = transitionMatrixCTB( n )
        return T @ c

    else:
        c[ 0 ] *= 2
        c[ -1 ] *= 2
        c *= n
        d = np.copy( c[ :n ] )

        coeffs = np.concatenate( c, d )

        samples = np.real( ifft( coeffs ) )

        return None



if __name__ == "__main__":

    pass

