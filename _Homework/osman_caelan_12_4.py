#osman_caelan_12_4.py

import numpy as np
from scipy import linalg as la
from autograd import elementwise_grad
from autograd import hessian
from autograd import numpy as anp


#problem 20
def newtons_method(f, x0, eps, M):
    #initialize variabels
    Df = elementwise_grad(f)
    Df_2 = hessian(f)
    #single input cases
    if anp.isscalar(x0):
        x = float(x0)
        for i in range(M):
            x_1 = x - (Df(x)/Df_2(x))
            #break if smaller than tolerance
            if abs(x-x_1) < eps:
                return x_1, True, i+1
            x = x_1
        return x, False, M

    #multiple input case
    else:
        x = x0.astype('float64')
        for i in range(M):
          if anp.isfinite(anp.linalg.cond(Df_2(x))):
            x_1 = x - anp.linalg.solve(Df_2(x), Df(x))
          else:
            #Levenberg-Marquardt Modification
            mu = min(la.eigvals(Df_2(x)))+ 1e-5
            x_1 =x - anp.linalg.solve(Df_2(x)+mu*anp.eye(len(x)), Df(x))
          #break if smaller than tolerance
          if la.norm(x_1 - x) < eps:
              return x_1, True, i+1
          x = x_1
        return x, False, M

#problem 21
def rosenbrock():
    x0 = anp.array([-2.,2.])
    rose = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    return newtons_method(rose, x0, 10e-5, 15)
