#change_of_basis.py

import numpy as np
from numpy.fft import fft
from scipy.special import binom
from matplotlib import pyplot as plt
import bernstein_functions as bf


def transitionMatrixBTC( n ):
    #define conjugate Kronecker delta function
    def cKronecker( delta ):
        if delta == 0:
            return 0
        else:
            return 1
    #construct transition matrix
    C = np.zeros( ( n+1, n+1 ) )
    for j in range( n+1 ):
        for k in range( n+ 1 ):
            C[ j, k ] = ( cKronecker( j ) + 1 ) / 4**( n + j )
            C[ j, k ] *= np.sum([ ( -1 )**( j+i )*binom( 2*j, 2*i)*
                                  binom( 2*( k+i ), k+i )*binom( 2*( n-k+j-i ), n-k+j+i)
                                  / binom( n + j , k + i)  for i in range( j + 1 ) ] )
    C
    return C


def bernsteinToChebyshev( b, domain = [0, 1], matrix=False ):
    '''Changes from the bernstein to chebyshev basis
       Paramaters:
           b ((n+1, ) ndarray): coordinates in the degree-n Bernstein basis
           matrix (bool): boolean to use fft or matrix transform
       Returns:
           coeffs ((n+1, ) ndarray): coordinates in the degree-n Chebyshev basis
    '''
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
        #construct transition matrix
        C = np.zeros( ( n+1, n+1 ) )

        for j in range( n+1 ):
            for k in range( n+ 1 ):
                C[ j, k ] = ( cKronecker( j ) + 1 ) / 4**( n + j )
                C[ j, k ] *= np.sum([ ( -1 )**( j+i )*binom( 2*j, 2*i)*
                                      binom( 2*( k+i ), k+i )*binom( 2*( n-k+j-i ), n-k+j+i)
                                      / binom( n + j , k + i)  for i in range( j + 1 ) ] )


        return C @ b


if __name__ == "__main__":

    #test for x^4 using fft
    x4 = np.zeros(5)
    x4[-1] = 1
    chebyFFT = bernsteinToChebyshev(x4)
    #test for x^4 using matrix
    chebyMatrix = bernsteinToChebyshev(x4, matrix=True)


    chebyBasis = [bf.chebyshevFunc(j) for j in range(5)]
    chebyBasis01 = [bf.chebyshevFunc(j, domain=[0, 1]) for j in range(5)]


    '''
    domain0 = np.linspace(0, 1, 100)
    print(chebyBasis[0](domain0))

    for func in chebyBasis01:
        plt.plot(domain0, func(domain0))

    plt.title('Chebyshev Basis domain=[0, 1]')
    plt.show()




    #plot fft
    '''







