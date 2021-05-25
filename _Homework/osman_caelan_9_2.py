#osman_caelan_9_2.py
import numpy as np
from matplotlib import pyplot as plt
import math

def barycentric_weights( x ):
    ''' Computes the barycentric weights for an interpolating polynomial
        Paramaters:
            x ( ( n+1, ) ndarray ): the x coordinates of the interpolation points
        returns:
            w ( ( n+1, ) ndarray ): the weights
    '''
    w = np.zeros( x.size )
    for i, point in enumerate( x ):
        w[ i ] = 1 / np.prod( [ point - x[ j ] for j in range( 0, x.size ) if j != i ] )

    return w


def evaluate_interpolated_function( x, y, x0 ):
    ''' creates an interpolating function from the points in (x, y) and evaluates the function at x0
        Paramaters: x ( ( n+1, ) ndarray ): the x coordinates of the interpolation points y ( ( n+1, ) ndarray ): the y coordinates of the interpolation points
            x0 ( float ): the point to evaluate the function at
       Returns:
           y0 ( float ): the evaluation point
    '''
    dictionary = { val : y[ i ] for i, val in enumerate( x ) }

    if type( x0 ) is float or type( x0 ) is int:
        if x0 in dictionary:
            y0 = dictionary[ x0 ]
            return y0
    else:
        w = barycentric_weights( x )
        y0 = np.zeros( x0.size )
        for i, val in enumerate( x0 ):
            if val in dictionary:
                y0[ i ] = dictionary[ val ]
            else:
                y0[ i ] = sum( [ w[ k ] / ( val - x[k] ) * y[ k ] for k in range( x.size ) ] )
                y0[ i ] /= sum( [ w[ k ] / ( val - x[ k ] ) for k in range( x.size ) ] )
        return y0

if __name__ == "__main__":
    '''
    func = lambda x: np.sin( np.pi * x )
    domain = np.linspace( -1, 1, 200 )
    #code for 25
    xvals_25 = np.array( [ -1, -1/3., 1/3., 1 ] )
    yvals_25 = np.array( [ np.sin( np.pi * x ) for x in xvals_25 ] )
    plt.plot( domain, func( domain ), 'k-', label='Sin(pix)' )
    plt.plot( domain, evaluate_interpolated_function( xvals_25, yvals_25, domain ), 'r--', label='Interpolation' )
    plt.legend( loc='best' )
    plt.title('Prob 25')
    plt.show()
    #code for 26
    xvals_26 = np.array( [ np.cos( j * np.pi / 3 ) for j in range( 4 ) ] )
    yvals_26 = np.array( [ np.sin( np.pi * x )  for x in xvals_26 ] )
    plt.plot( domain, func( domain ), 'k-', label='Sin(pix)' )
    plt.plot( domain, evaluate_interpolated_function( xvals_26, yvals_26, domain ), 'r--', label='Interpolation' )
    plt.legend( loc='best' )
    plt.title('Prob 26')
    plt.show()
    '''

    #code for 28
    coordinates = [ 1, 20 ]
    degree = 20
    change_variables = lambda x: ( ( -coordinates[ 0 ] + coordinates[ 1 ] ) / 2. ) * x + ( coordinates[ 0 ] + coordinates[ 1 ] ) / 2
    cheby_zeros = np.array( [ np.cos( np.pi * ( j +  1 /2.  ) / degree ) for j in range( degree ) ] )
    shifted_cheby_zeros = change_variables( cheby_zeros )

    #define W and q
    W = lambda x: np.prod( np.array( [ x - i for i in range( 1, 21 ) ] ) )
    q = lambda x: np.prod( np.array( [ x - z for z in shifted_cheby_zeros ] ) )
    domain = np.linspace( coordinates[ 0 ], coordinates[ 1 ], 500 )

    rnge_W = np.zeros( domain.size )
    rnge_q = np.zeros( domain.size )

    for i, x in enumerate( domain ):
        rnge_W[ i ] = W( x )
        rnge_q[ i ] = q( x )

    difference = np.abs( rnge_W - rnge_q )
    sup = np.max( difference )
    sup_W = np.max( rnge_W )
    sup_q = np.max( rnge_q )

    plt.plot( domain, rnge_W, label='Wilkinson' )
    plt.plot( domain, rnge_q, label='q' )
    #plt.suptitle( 'Max W: ' + str( sup_W ) + '\nMax q: ' + str( sup_q ) )
    plt.title( r'$||f(x) - p(x)||_{L_{\infty}}$ Difference: ' + str( math.trunc( sup ) ) )
    plt.legend( loc='best' )
    plt.ylim( -5e13, 5e13 )
    plt.xlim( 1, 20 )
    plt.show()




