import numpy as np
from numpy.linalg import inv, norm


def BFGS(f, df, x0, A0, max_iter=40, tol=1e-8):
    """Minimize f using BFGS, given the derivative df, an
    initial guess x0, and an initial approx A0 of D^2f(x0).
    """

    #initialize
    done = False
    iters = 0 #count the number of iterations
    A_inv = inv(A0) #initial approximate inverse hessian
    x = x0 - A_inv @ df(x0) #x_1
    s = x - x0 #s_1


    while not done: #Main BFGS loop
        y = df(x) - df(x0) #update y
        sy = s @ y
        Ay = A_inv @ y
        #approxmiate the new inverse hessian
        A_inv = (A_inv + ((sy + y @ Ay) / sy**2) * np.outer(s, s)
                 - (np.outer(Ay, s) + np.outer(s, Ay)) / sy)

        x0 = x
        x = x0 - A_inv @ df(x0) #update x
        s = x - x0 #update s
        iters += 1
        #stopping criteria
        done = ((norm(s) < tol) or
                (norm(df(x)) < tol) or
                (np.abs(f(x) - f(x0)) < tol) or
                (iters >= max_iter))


    return x, iters


def prob12_30(part = 1):

    if part == 1:
        fx = lambda x: x[0]**3 - 3*x[0]**2 + x[1]**2
        df = lambda x: np.array([3*x[0]**2 - 6*x[0], 2*x[1]])
        x0 = np.array([4, 4])
        A0 = np.array([[18, 0], [0, 2]])
        final, num_iterations = BFGS(fx, df, x0, A0, tol=1e-5)
        return final, num_iterations

    elif part == 2:
        fx = lambda x: x[0]**3 - 3*x[0]**2 + x[1]**2
        df = lambda x: np.array([3*x[0]**2 - 6*x[0], 2*x[1]])
        x0 = np.array([4, 4])
        A0 = np.eye(2)
        final, num_iterations = BFGS(fx, df, x0, A0, tol=1e-5)
        return final, num_iterations

    elif part == 3:
        fx = lambda x: x[0]**3 - 3*x[0]**2 + x[1]**2
        df = lambda x: np.array([3*x[0]**2 - 6*x[0], 2*x[1]])
        x0 = np.array([10, 10])
        A0 = np.array([[54, 0], [0, 2]])
        final, num_iterations = BFGS(fx, df, x0, A0, tol=1e-5)
        return final, num_iterations

    elif part == 4:
        fx = lambda x: x[0]**3 - 3*x[0]**2 + x[1]**2
        df = lambda x: np.array([3*x[0]**2 - 6*x[0], 2*x[1]])
        x0 = np.array([10, 10])
        A0 = np.eye(2)
        final, num_iterations = BFGS(fx, df, x0, A0, tol=1e-5)
        return final, num_iterations

    elif part == 5:
        print("The problem with the point x0 = [0, 0] is that the innter product of <sk, yk> will return 0 as it's value. \nSo A_inv can not be calculated for the next step. \nThis happens because [0, 0] is a critical point of the funcion.")

    else:
        raise NotImplementedError

