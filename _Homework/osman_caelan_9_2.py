#osman_caelan_9_2.py

import numpy as np
from matplotlib import pyplot as plt

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
        Paramaters:
            x ( ( n+1, ) ndarray ): the x coordinates of the interpolation points
            y ( ( n+1, ) ndarray ): the y coordinates of the interpolation points
            x0 ( float ): the point to evaluate the function at
       Returns:
           y0 ( float ): the evaluation point
    '''
    w = barycentric_weights( x )
    y0 = sum( [ w[ j ] / ( x0 - x[ j ] ) * y[ j ] for j in range( x.size ) ] )
    y0 /= sum( [ w[ j ] / ( x0 - x[ j ] ) for j in range( x.size ) ] )

    return y0

'''
if __name__ == "__main__":

    count = 0
    fig, axs = plt.subplots( 2, 3 )
    n = [ 2, 3, 10, 11, 19, 20 ]

    domain1 = np.linspace( -1, 1, 200 )
    rnge = np.absolute( domain1 )
    for i in range( 2 ):
        for j in range( 3 ):
            domain = [-1 + n / 100 for n in range(201)]
            x = np.linspace( -1, 1, n[ count ] )
            for k, point in enumerate( x ):
                if point in domain:
                    domain.remove( point )
            domain.insert( 0, -1.0001 )
            domain.append( 1.0001 )
            y = np.absolute( x )
            output = evaluate_interpolated_function( x, y, np.array( domain ) )
            ax = axs[ i, j ]
            ax.plot( domain1, rnge, 'r', label='f' )
            ax.plot( domain, output, 'k', label = 'n = ' + str( n[ count ] ) )
            ax.legend(loc='best')
            count += 1

    plt.show()
'''

