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

    assert np.allclose( cM1, cob.transitionMatrixBTC( 1 ) ), "Testing correct n=1 BTC transition."
    assert np.allclose( cM2, cob.transitionMatrixBTC( 2 ) ), "Testing correct n=2 BTC transition."
    assert np.allclose( cM3, cob.transitionMatrixBTC( 3 ) ), "Testing correct n=3 BTC transition."

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

    assert np.allclose( cIM1, cob.transitionMatrixCTB( 1 ) ), "Testing correct n=1 BTC transition."
    assert np.allclose( cIM2, cob.transitionMatrixCTB( 2 ) ), "Testing correct n=2 BTC transition."
    assert np.allclose( cIM3, cob.transitionMatrixCTB( 3 ) ), "Testing correct n=3 BTC transition."

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

    #test bernstein basis
    bernsteinBasis = [ bf.bernsteinFunc( 4, i ) for i in range( 5 ) ]

    #f(x) = x^4
    monomialX4 = lambda x: x**4
    bernsteinX4 = lambda x: np.sum([ coordX4[ i ] * f( x ) for i, f in enumerate( bernsteinBasis ) ])
    output1 = monomialX4( domain )
    output2 = [ bernsteinX4( x ) for x in domain ]

    assert np.allclose( output1, output2 ), "Testing Bernstein basis creation x^4."

    #f(x) = 1 - 4x^3+5x^4
    monomial5X4 = lambda x: 1 - 4*x**3 + 5*x**4
    bernstein5X4 = lambda x: np.sum([ coord5X4[ i ] * f( x ) for i, f in enumerate( bernsteinBasis ) ])
    output3 = monomial5X4( domain )
    output4 = [ bernstein5X4( x ) for x in domain ]

    assert np.allclose( output3, output4 ), "Testing Bernstein basis creation 1-4x^3+5x^4."

    correctFFT = np.array([ 3/8., 0, 1/2., 0, 1/8. ])
    correctMatrix = np.array([ 0.2734375, 0.4375, 0.21875, 0.0625, 0.0078125 ])
    assert np.allclose( cob.bernsteinToChebyshev( coordX4 ), correctFFT ), "Testing FFT on x^4."
    assert np.allclose( cob.bernsteinToChebyshev(coordX4, matrix=True ), correctMatrix ), "Testing matrix on x^4."

    #1-4x^3 + 5x^4
    correctFFT = np.array([ 2.875, -3.,  2.5, -1., 0.625 ])
    correctMatrix = np.array([ 1.1171875, 0.3125, 0.34375, 0.1875, 0.0390625 ])
    assert np.allclose( cob.bernsteinToChebyshev( coord5X4 ), correctFFT ), "Testing FFT on 1-4x^3+5x^4."
    assert np.allclose( cob.bernsteinToChebyshev( coord5X4, matrix=True ), correctMatrix ), "Testing matrix on 1-4x^3+5x^4."


def test_plot( create_arrays ):

    coordX4, coord5X4 = create_arrays

    chebyBasis = [ bf.chebyshevFunc( j ) for j in range( 5 ) ]
    chebyBasis01 = [ bf.chebyshevFunc( j, domain=[ 0, 1 ] ) for j in range( 5 ) ]

    domain = np.linspace( 0, 1, 100 )

    coordFFTX4 = cob.bernsteinToChebyshev( coordX4 )
    coordMatrixX4 = cob.bernsteinToChebyshev( coordX4, matrix=True )

    x4 = lambda x: x**4
    x4FFT = lambda x: np.sum([ coordFFTX4[ i ] * f( x ) for i, f in enumerate( chebyBasis ) ])
    x4Matrix = lambda x: np.sum([ coordMatrixX4[ i ] * f( x ) for i, f in enumerate( chebyBasis01 ) ])

    x4Output = x4( domain )
    x4FFTOutput = [ x4FFT( x ) for x in domain ]
    x4MatrixOutput = [ x4Matrix( x ) for x in domain ]

    coordFFT5X4 = cob.bernsteinToChebyshev( coord5X4 )
    coordMatrix5X4 = cob.bernsteinToChebyshev( coord5X4, matrix=True )

    fiveX4 = lambda x: 1 - 4*x**3 + 5*x**4
    fiveX4FFT = lambda x: np.sum([ coordFFT5X4[ i ] * f( x ) for i, f in enumerate( chebyBasis ) ])
    fiveX4Matrix = lambda x: np.sum([ coordMatrix5X4[ i ] * f( x ) for i, f in enumerate( chebyBasis01 ) ])

    fiveX4Output = fiveX4( domain )
    fiveX4FFTOutput = [ fiveX4FFT( x ) for x in domain ]
    fiveX4MatrixOutput =  [ fiveX4Matrix( x ) for x in domain ]

    assert np.allclose( x4Output, x4FFTOutput )
    assert np.allclose( x4Output, x4MatrixOutput )
    assert np.allclose( x4FFTOutput, x4MatrixOutput )

    assert np.allclose( fiveX4Output, fiveX4FFTOutput )
    assert np.allclose( fiveX4Output, fiveX4MatrixOutput )
    assert np.allclose( fiveX4FFTOutput, fiveX4MatrixOutput )

    ax1 = plt.subplot(121)
    ax1.plot( domain, x4Output, 'b', linewidth=8, label=r'$x^4$')
    ax1.plot( domain, x4FFTOutput, 'r', linewidth=3.5, label='FFT')
    ax1.plot( domain, x4MatrixOutput, 'k', linewidth=1, label='Matrix Transform')
    ax1.legend(loc='best')

    ax2 = plt.subplot(122)
    ax2.plot( domain, fiveX4Output, 'b', linewidth=8, label=r'$1-4x^3+5x^4$')
    ax2.plot( domain, fiveX4FFTOutput, 'r', linewidth=3.5, label='FFT')
    ax2.plot( domain, fiveX4MatrixOutput, 'k', linewidth=1, label='Matrix Transform')
    ax2.legend(loc='best')
    plt.suptitle(r'Basis transformation for $x^4$ and $1-4x^3 + 5x^4$')

    plt.show()


if __name__ == "__main__":

    pass
