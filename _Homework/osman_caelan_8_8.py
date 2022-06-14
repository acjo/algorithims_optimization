#osman_caelan_8_8.py

import numpy as np
from matplotlib import pyplot as plt

def sampling(f, n):

    points = np.array([k/(2**n) for k in range(2**n)])

    return f(points)


def haar_wavelets(a, x):

    #compute n, n will be an integer
    n = int(np.log2(len(a)))
    #compute the evaluation points
    evals = np.array([coef for k, coef in enumerate(a) if k / (2 ** n ) <= x and x < ( k + 1 ) / (2 ** n ) ] )
    #sum and return
    return sum(evals)


def exercise8_42( ):

    def func( t ):
        return np.sin( ( 2 * np.pi * t ) - 5 ) / np.sqrt( abs( t - ( np.pi / 20  ) ) )

    n = np.arange(1, 11)

    fig, axs = plt.subplots( 2, 5 )
    domain = np.linspace( 0, 1, 100, endpoint=False )
    i = 0
    for j in range( 2 ):
        for k in range( 5 ):
            ax = axs[ j, k ]
            ax.plot( domain, func( domain ), label='original' )
            output = []
            for x in domain:
                output.append( haar_wavelets( sampling( func, n[ i ] ), x ) )
            ax.plot( domain, np.array( output ), label = 'fn' )
            ax.set_title( 'n = ' + str( n[ i ] ) )
            ax.legend(loc='best')
            i += 1

    plt.show()


if __name__ == "__main__":

    def Haar_Father(t):

        return np.where((0 <= t) & ( t < 1), 1, 0)

    domain = np.linspace(-2, 2, 200) 

    print(Haar_Father(1))

    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.plot(domain, Haar_Father(domain))
    plt.show()
