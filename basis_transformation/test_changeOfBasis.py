#test_changeOfBasis.py

import pytest
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import basisFunctions as bf
import changeOfBasis as cob


def test_cKronecker():
    assert cob.cKronecker( 0 ) == 0
    assert cob.cKronecker( 1e-15 ) == 1

def test_transitionMatrixBTC():
    cM1 = np.array([[ 1/2., 1/2. ],
                    [ -1/2., 1/2. ]])

    cM2 = np.array([[ 3/8., 1/4., 3/8. ],
                    [ -1/2., 0, 1/2. ],
                    [ 1/8., -1/4., 1/8. ]])

    cM3 = np.array([[ 5/16., 3/16., 3/16., 5/16. ],
                    [ -15/32, -3/32, 3/32, 15/32 ],
                    [ 3/16, -3/16, -3/16, 3/16 ],
                    [ -1/32., 3/32, -3/32, 1/32 ]])

    assert np.allclose( cM1, cob.transitionMatrixBTC( 1 ) )
    assert np.allclose( cM2, cob.transitionMatrixBTC( 2 ) )
    assert np.allclose( cM3, cob.transitionMatrixBTC( 3 ) )

def test_transitionMatrixCTB():
    cM1 = np.array([[ 1/2., 1/2. ],
                    [ -1/2., 1/2. ]])

    cM2 = np.array([[ 3/8., 1/4., 3/8. ],
                    [ -1/2., 0, 1/2. ],
                    [ 1/8., -1/4., 1/8. ]])

    cM3 = np.array([[ 5/16., 3/16., 3/16., 5/16. ],
                    [ -15/32, -3/32, 3/32, 15/32 ],
                    [ 3/16, -3/16, -3/16, 3/16 ],
                    [ -1/32., 3/32, -3/32, 1/32 ]])

    cIM1 = la.inv( cM1 )
    cIM2 = la.inv( cM2 )
    cIM3 = la.inv( cM3 )

    assert np.allclose( cIM1, cob.transitionMatrixCTB( 1 ) )
    assert np.allclose( cIM2, cob.transitionMatrixCTB( 2 ) )
    assert np.allclose( cIM3, cob.transitionMatrixCTB( 3 ) )

def test_exceptions():
    #basisFunctions
    pytest.raises( ValueError, bf.bernsteinFunc, n=3, k=4 )
    pytest.raises( TypeError, bf.bernsteinFunc, n=4, k=3, domain = 'hi' )
    pytest.raises( TypeError, bf.bernsteinFunc, n=4, k=3, domain = [1, 2, 3] )
    pytest.raises( ValueError, bf.bernsteinFunc, n=4, k=3, domain = [3, 1] )

    pytest.raises( TypeError, bf.chebyshevFunc, n=3, domain='hi' )
    pytest.raises( TypeError, bf.chebyshevFunc, n=3, domain=[1, 2, 3] )
    pytest.raises( ValueError, bf.chebyshevFunc, n=3, domain=[3, 1] )
    pytest.raises( ValueError, bf.chebyshevFunc, n=-1, domain=[0, 1] )

    #changeOfBasis
    pytest.raises( TypeError, cob.bernsteinToChebyshev, b=3 )
    b = np.array([ 4, 3, 5 ])
    pytest.raises( TypeError, cob.bernsteinToChebyshev, b=b, domain=6 )
    pytest.raises( TypeError, cob.bernsteinToChebyshev, b=b, domain=[1, 2, 3] )
    pytest.raises( ValueError, cob.bernsteinToChebyshev, b=b, domain=[3, 1] )
    pytest.raises( TypeError, cob.bernsteinToChebyshev, b=b, domain=[0, 1], matrix=list() )

    pytest.raises( TypeError, cob.transitionMatrixBTC, n=3.4 )

    pytest.raises( TypeError, cob.transitionMatrixCTB, n=3.4 )

@pytest.fixture
def create_arrays():
    x4 = np.zeros( 5 )
    x4[ -1 ] = 1
    fiveX4 = np.array([ 1, 1, 1, 0, 2 ])

    return x4, fiveX4



def test_bernsteinToChebyshev( create_arrays ):
    coordX4, coord5X4 = create_arrays
    domain = np.linspace( 0, 1, 100)

    print(coord5X4)
    #test bernstein basis
    bernsteinBasis = [ bf.bernsteinFunc( 4, i ) for i in range( 5 ) ]

    #f(x) = x^4
    monomialX4 = lambda x: x**4
    bernsteinX4 = lambda x: np.sum([ coordX4[ i ] * f( x ) for i, f in enumerate( bernsteinBasis ) ])
    output1 = monomialX4( domain )
    output2 = [ bernsteinX4( x ) for x in domain ]

    ax1 = plt.subplot( 121 )
    ax1.plot( domain, output1, label='monomial' )
    ax1.plot( domain, output2, label='bb' )
    ax1.legend( loc='best' )

    assert np.allclose( output1, output2 ), "testing Bernstein basis creation x^4"

    #f(x) = 1 - 4x^3+5x^4
    monomial5X4 = lambda x: 1 - 4*x**3 + 5*x**4
    bernstein5X4 = lambda x: np.sum([ coord5X4[ i ] * f( x ) for i, f in enumerate( bernsteinBasis ) ])
    output3 = monomial5X4( domain )
    output4 = [ bernstein5X4( x ) for x in domain ]

    ax2 = plt.subplot( 122 )
    ax2.plot( domain, output3, label='monomial' )
    ax2.plot( domain, output4, label='bb' )
    ax2.legend( loc='best' )

    plt.show()

    #assert np.allclose( output3, output4 ), "testing Bernstein basis creation 1-4x^3+5x^4"




    correctFFT = np.array([ 3/8., 0, 1/2., 0, 1/8. ])
    correctMatrix = np.array([ 0.2734375, 0.4375, 0.21875, 0.0625, 0.0078125 ])
    assert np.allclose( cob.bernsteinToChebyshev( coordX4 ), correctFFT )
    assert np.allclose( cob.bernsteinToChebyshev(coordX4, matrix=True ), correctMatrix )

    #1-4x^3 + 5x^4
    correctFFT = np.array([ 17.875, -29., 18.5, -7., 1.625 ])
    correctMatrix = np.array([ 1.1171875, 0.3125, 0.34375, 0.1875, 0.0390625 ])
    assert np.allclose( cob.bernsteinToChebyshev( coord5X4 ), correctFFT )
    assert np.allclose( cob.bernsteinToChebyshev( coord5X4, matrix=True ), correctMatrix )











'''
def test_plot(test='test1'):

    if test not in [ 'test1', 'test2' ]:
        raise ValueError ( 'Wrong test deliminator' )
    chebyBasis = [ bf.chebyshevFunc( j ) for j in range( 5 ) ]
    chebyBasis01 = [ bf.chebyshevFunc( j, domain=[ 0, 1 ] ) for j in range( 5 ) ]

    domain = np.linspace( 0, 1, 100 )
    #domain2 = np.linspace( -1, 1, 100 )

    if test == 'test1':
        x4 = np.zeros( 5 )
        x4[ -1 ] = 1
        chebyFFT = cob.bernsteinToChebyshev( x4 )
        #test for x^4 using matrix
        chebyMatrix = cob.bernsteinToChebyshev( x4, matrix=True )
        monomial = lambda x: x**4

        #function from fft
        x4FFT = lambda x : np.sum([ chebyFFT[ i ] * f( x ) for i, f in enumerate( chebyBasis ) ])
        #function from matrix
        x4Matrix = lambda x : np.sum([ chebyMatrix[ i ] * f( x ) for i, f in enumerate( chebyBasis01 ) ])
        ax = plt.axes()
        ax.set_facecolor( 'gray' )

        outputFFT = np.array([ x4FFT( x ) for x in domain ])
        outputMatrix = np.array([ x4Matrix( x ) for x in domain ])
        plt.plot( domain, monomial( domain ), 'w', linewidth=7, label='original in monomial basis' )
        plt.plot( domain, outputFFT, 'b', linewidth=3, label='COB using FFT' )
        plt.plot( domain, outputMatrix, 'r', linewidth=1, label='COB using Matrix' )
        plt.legend( loc='best' )
        plt.title( r'$x^4$ in different polynomial bases' )
        plt.show()

    elif test == 'test2':

        monomial = lambda x: 1 - 4*x**3 + 5*x**4
        bernstein = np.array([ 1, 1, 1, 0, 2 ])


        chebyFFT = cob.bernsteinToChebyshev( bernstein )
        chebyMatrix = cob.bernsteinToChebyshev( bernstein, matrix=True )

        funcFFT = lambda x: np.sum([ chebyFFT[ i ] * f( x ) for i, f in enumerate( chebyBasis ) ])
        funcMatrix = lambda x: np.sum([ chebyMatrix[ i ] * f( x ) for i, f in enumerate( chebyBasis01 ) ])

        outputFFT = np.array([ funcFFT( x ) for x in domain ])
        outputMatrix = np.array([ funcMatrix( x ) for x in domain ])

        plt.plot( domain, monomial( domain ), 'b', linewidth=8, label='original in monomial basis' )
        plt.plot( domain, outputFFT, 'r', linewidth=3.5, label='C.O.B. using FFT' )
        plt.plot( domain, outputMatrix, 'k', linewidth=1, label='C.O.B. using matrix' )
        plt.legend( loc='best' )
        plt.title( r'$1-4x^3+5x^4$ in different polynomial bases' )
        plt.show()
'''


if __name__ == "__main__":

    #create_arrays = (np.array([0, 0, 0, 0, 1]), np.array([1, 1, 1, 0, 2]))
    #test_bernsteinToChebyshev( create_arrays )
    pass
