#osman_caelan_8_9.py
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la

def FWT(a, j=0):
    '''
    testing
    Haar FWT of "a" down to level j
    paramaters:
    a (2**k, 1) ndarray
    j integer
    returns:
    the wavelet transform
    a(updated), L[::-1]
    '''
    #assume len(a) is an integer power of 2
    m = int(np.log2(len(a)))
    #list of partial transforms
    L = list()

    while m > j:
        b_m_1 = 0.5 * (a[::2] -a[1::2])
        L.append(0.5 * (a[::2] - a[1::2]))
        a = 0.5 * (a[::2] + a[1::2])
        m -= 1

    return a, L[::-1]

def sampling( f, n ):
    ''' takes a function and samples f on those points
        Paramaters:
        f: lambda function
        n: integer
        returns:
        ((2**n)) ndarray of the points
        ((2**n,)) ndarray of the function evaluation points
    '''
    points = np.array( [ k / ( 2**n ) for k in range( 2**n ) ] )
    return points, f( points )


def haar_coefficients( a, x ):
    '''
    paramaters:
    a: ((2**n,)) ndarray containing evaluation points of a function
    x: a point in the domain of the function determing the coeffcient
       of the given father/son function
    returns:
    ((2**n,)) ndarray containing coefficients
    '''
    #compute n, n will be an integer
    n = int( np.log2( len( a ) ) )
    #compute the valuation points
    evals = np.array( [ coef for k, coef in enumerate( a ) if k / (2 ** n ) <= x and x < ( k + 1 ) / (2 ** n ) ] )
    #sum and return
    return evals


def w_j_basis(j, k, father=False):
    ''' Computes the father/mother/daughter basis of W_j
        Paramaters:
        j: int
        k: int
        father: boolean (optional)
        returns:
        function object (father/mother daughter)
    '''
    if father:
        def father(x):
            ''' The father function
            '''
            if 0 <= x and x < 1:
                return 1
            else:
                return 0
        return father

    else:
        def daughter(x):
            ''' The mother/daughter function
            '''
            if k/2**j <=x and x < (2*k+1)/(2**(j+1)):
                return 1
            elif (2*k+1)/(2**(j+1)) <=x and x < (k+1)/(2**j):
                return -1
            else:
                return 0
        return daughter


def v_j_basis(j, k):
    ''' Computes the father/son basis of V_j
        Paramaters:
        j: int
        k: int
        returns:
        function object (father/song)
    '''

    def son(x):
        '''The father/son function
        '''
        if k / 2**j <= x and x < (k+1) / 2**j:
            return 1
        else:
            return 0
    return son

def IWM(basis, j):
    ''' algorthimically assembles the inverse wavelet transform matrix
        Paramaters:
        basis: ((2**j,)) ndarray containing basis function objects
        int: j the dimension of the R^(2**j)
    '''
    dim = 2**j
    sampling = np.linspace(0, 1, dim, endpoint=False)
    evals = []
    for func in basis:
        func_eval = []
        for sample in sampling:
            func_eval.append(func(sample))
        evals.append(func_eval)

    H = np.zeros((dim, dim))
    for c in range(H.shape[1]):
        H[:, c] = evals[c]

    return H

def approximate(f, basis):
    ''' approximates a function in a given basis
    '''
    def func(x):
        return sum([f[k] * basis[k](x) for k in range(len(f))])
    return func

def create_wj_basis(j):
    ''' creates an array containing the wj_basis
        paramaters:
        j: int
        returns:
        ((2**j)) ndarray of basis function objects
    '''
    basis = []
    #append father
    basis.append(w_j_basis(0, 0, True))
    #append mother
    basis.append(w_j_basis(0, 0))
    #append daughters
    for i in range(1, j):
        for k in range(2**i):
            basis.append(w_j_basis(i,k))

    return basis

def get_b(a, j=0):
    ''' gets the coefficients for the detail function
    '''

    m = int(np.log2(len(a)))
    L = []
    while m > j:
        L.append(0.5 * (a[::2] - a[1::2]))
        a = 0.5 * (a[::2] + a[1::2])
        m -= 1

    return L

def prob8_45(wavelet, j):

    ''' takes a wavelet and an integer j computes the inverse wavelet
        transform and returns a function in V_j that approximates f and a function
        g in V_j perp  that describes the detail
        paramaters:
        wavelet ((2^j,)) ndarray
        j integer
        returns:
        f lambda (approximation)
        g lambda (detail)
    '''

    #dimension of R^n (number of basis elements)
    dim = 2 ** j
    #sample points
    sampling = np.linspace(0, 1, dim, endpoint=False)
    wj_basis = create_wj_basis(j)
    #get inverse wavelet matrix
    H_n = IWM(wj_basis, j)
    #get coordinate vector
    f_n = H_n @ wavelet
    b_n = get_b(f_n, j)
    b = []
    #get b coefficients
    for k, mat in enumerate(b_n):
        for row in mat:
            for col in row:
                b.append(col)
    '''
    for _ in range(len(f_n) - len(b)):
        b.insert(0, 0)
    '''
    #get son/father basis
    vj_basis = [v_j_basis(j, k) for k in range(0, 2**j)]
    #get projection of function onto Vj
    T_j = approximate(f_n, vj_basis)
    g_j = approximate(b, wj_basis)
    return T_j, g_j

def prob_8_45_3():


    js = [2, 4, 6]
    fig, axs = plt.subplots(1, 3)
    function = lambda x: 100 * (x**2) * (1 - x) * abs(np.sin( 10 * x/ 3))
    domain = np.linspace(0, 1, 100, endpoint=False)

    for k, j in enumerate(js):
        ax = axs[k]
        #get sample points and evalution points on samples
        sample_points, evals = sampling(function, j)
        #get the current basis for the basis Wj
        current_basis = create_wj_basis(j)
        #compute the coefficeints in terms of the vj basis
        coefficients = np.array([haar_coefficients(evals, x) for x in sample_points])
        #compute the wavelet transform
        wavelet = la.inv(IWM(current_basis, j)) @ coefficients
        #get the projection
        projection, gj = prob8_45(wavelet, j)
        #plot function and projection on domain
        ax.plot(domain, function(domain), 'k', label='Original')
        evaluation = np.array([projection(x) for x in domain])
        #evaluation_c = np.array([projection(x) + gj(x) for x in domain])
        ax.plot(domain, evaluation, 'r', label='Approximation')
        #ax.plot(domain, evaluation_c, 'c', label='combined')
        ax.set_title('j = '+ str(j))
        ax.legend(loc='best')

    plt.show()

    return
