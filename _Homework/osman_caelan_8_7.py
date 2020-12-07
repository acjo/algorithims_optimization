#osman_caelan_8_7.py
import numpy as np
from matplotlib import pyplot as plt


#helper function, adds two lambda functions
def f_add(f1, f2):
    '''Adds two lambda functions together
       Paramaters:
       (lambda) f1, f2
       Returns:
       (lambda) f1 + f2
    '''
    return lambda *args, **kwds: f1(*args, **kwds) + f2(*args, **kwds)

#helper function perfoms DFT
def DFT(f):
    ''' Computes the discrete Fourier Transform of the 1d array f
    '''
    n = len( f )
    m = np.arange( n ).reshape( n, 1 )
    W = np.exp( ( -2j * np.pi/n ) * m @ m.T )
    return W @ f / n

#helper function performs FFT
def FFT(f):
    ''' Performs the FFT algorithm on the numpy array "f"
    '''
    n = len( f )

    if n <= 4:
        return DFT( f )
    else:
        f_even = FFT( f[ ::2 ] )
        f_odd = FFT( f[ 1::2 ] )
        w = np.exp( ( -2j * np.pi / n ) * np.arange( n ) )
        first_sum = f_even + w[ :n//2 ] * f_odd
        second_sum = f_even + w [ n//2: ] * f_odd
        return 0.5 * np.concatenate( [ first_sum, second_sum ] )


def periodic_sampling_theorem(nu, n, T, f, g):

    omega = 2 * np.pi / T
    domain = np.arange( 0, T, T / n ) #get sample points
    f_vec = f( domain ) #compute vector from the function f
    f_hat = DFT( f_vec ) #compute the FFT of f_vec
    #compute coefficients
    coef = []
    for k in range( -nu, nu + 1 ):
        if -nu <= k and k < 0:
            coef.append( f_hat[ k + n ] )
        else:
            coef.append( f_hat[ k ] )

    coef = np.array(coef)

    if g:
        #compute g
        def func( t ):
            return sum( [ coef[ k + nu ] * np.exp( 1j * omega * k * t) for k in range( -nu, nu + 1 ) ] )
        return func

    #otherwise return the coefficients
    else:
        return coef

def prob34():

    def approximate( n, f):

        T = 1
        nu = n // 2
        domain = np.arange( 0, T, T / n )
        sample = f( domain )

        return sample, periodic_sampling_theorem(nu, n, T, f, True)

    fig, axs = plt.subplots( 2, 2 )

    n = [ 3, 7, 11, 13 ]

    i = 0

    f = lambda x: 1 - 3 * np.sin( 12 * np.pi * x + 7 ) + 5 * np.sin( 2 * np.pi * x - 1 ) + 5 * np.sin( 4 * np.pi * x - 3 )
    domain = np.linspace( 0, 1, 200 )

    for j in range( 2 ):
        for k in range( 2 ):
            ax = axs[ j, k ]
            points = np.arange( 0, 1, 1 / n[ i ])
            sample, g_n = approximate( n[ i ] ,f )
            ax.plot( points, sample, 'c.', label='samples')
            ax.plot(domain, f(domain), label = 'original')
            ax.plot(domain, g_n(domain), label='g_n')
            ax.set_title('n = ' + str(n[i]))
            ax.legend(loc='best')
            i += 1

    plt.show()


prob34()
