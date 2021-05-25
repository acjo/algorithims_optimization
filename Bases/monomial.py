#monomial.py
import numpy as np
from matplotlib import pyplot as plt

def monomialGenerator( i ):

    def monomial( x ):
        return x**i

    return monomial


class monomialBasis ( object ):

    def __init__( self, n ):

        if type( n ) is not int:
            raise TypeError( "constructor needs an integer input." )

        self.n = n

        self.functions = [ monomialGenerator( i ) for i in range( n+1 ) ]


    def plot( self, numPoints, individual=True, power = 0, domain=[ 0, 1 ] ):
        if type( individual ) is not bool:
            raise TypeError( "Individual needs to be a boolean." )
        if type(numPoints) is not int or numPoints <= 0:
            raise TypeError( "Number of points needs to be a positive integer." )
        if type( domain ) is not list:
            raise TypeError( "domain needs to be a list." )

        points = np.linspace( domain[ 0 ], domain[ -1 ], numPoints )
        if individual:
            output = self.functions[ power ]( points )
            plt.plot(points, output, label=r'$x^{}$'.format( power ) )
            plt.legend( loc='best' )
            plt.show()


        else:
            outputs = np.array([ [ function(x) for x in points ] for function in self.functions ] )
            for i in range( self.n + 1 ):
                plt.plot( points, outputs[ i ], label=r'$x^{{{}}}$'.format( i ) )
                plt.legend(loc='best')

            plt.suptitle(r"Monomial Basis for $\mathcal{P}_{10}$")
            plt.show()






if __name__ == "__main__":

    basis = monomialBasis( 10 )
    basis.plot( 150, individual=False )


