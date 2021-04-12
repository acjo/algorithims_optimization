import time
import numpy as np
from autograd import grad
from scipy import linalg as la
from matplotlib import pyplot as plt
from autograd import numpy as anp

def prob18():
    #submit pictures of plots and code on gradescope
    x0 = 1.5
    k = np.arange(1,54)
    h_list = 1/2**(1*k)


    # initialize arrays
    # accuracy
    acc_fd = []
    acc_cd = []
    acc_complex = []
    acc_auto = []
    # convergence rate
    conv_fd = []
    conv_cd = []
    conv_complex = []
    # time
    time_fd = []
    time_cd = []
    time_complex = []
    time_auto = []
 

    # set the functions
    f= lambda x: (anp.sin(x)**3 + anp.cos(x)) / anp.exp(x)
    Df = lambda x: -np.exp(-x)*(np.sin(x)**3 - 3*np.cos(x)*np.sin(x)**2 + np.sin(x) + np.cos(x))

    def fdq1(f, x, h=1e-5):
        """forward difference quotient"""
        return (f(x+h) - f(x)) / h
    def cdq2(f, x, h=1e-5):
        """centered difference quotient"""
        return (f(x+h) - f(x-h)) / (2*h)
    def csd(f,x,h=1e-5):
        """complex difference quotient"""
        return np.imag(f(x + 1j*h) / h)

    exact = Df(x0)

    for i, h in enumerate(h_list):

        start = time.time()
        forward = fdq1(f, x0, h)
        time_fd.append(time.time() - start)
        start = time.time()
        centered = cdq2(f, x0, h)
        time_cd.append(time.time() - start)
        start = time.time()
        complex_step = csd(f, x0, h)
        time_complex.append(time.time() - start)
        start = time.time()
        autoDf = grad(f)(float(x0))
        time_auto.append(time.time() - start)
        acc_fd.append(1 - la.norm(exact - forward) / la.norm(exact))
        acc_cd.append(1 - la.norm(exact - centered) / la.norm(exact))
        acc_complex.append(1 - la.norm(exact - complex_step) / la.norm(exact))
        acc_auto.append(1 - la.norm(exact - autoDf) / la.norm(exact))
        if i != 0:
            conv_fd.append(abs((abs(fdq1(f, x0, h_list[i]) - exact) - abs(fdq1(f, x0, h_list[i-1]) - exact)) / abs(k[i] - k[i-1])))
            conv_cd.append(abs((abs(cdq2(f, x0, h_list[i]) - exact) - abs(cdq2(f, x0, h_list[i-1]) - exact)) / abs(k[i] - k[i-1])))
            conv_complex.append(abs((abs(csd(f, x0, h_list[i]) - exact) - abs(csd(f, x0, h_list[i-1]) - exact)) / abs(k[i] - k[i-1])))
        else:
            conv_fd.append(abs((abs(fdq1(f, x0, h_list[i]) - exact) - abs(fdq1(f, x0, h_list[i+1]) - exact)) / abs(k[i] - k[i+1])))
            conv_cd.append(abs((abs(cdq2(f, x0, h_list[i]) - exact) - abs(cdq2(f, x0, h_list[i+1]) - exact)) / abs(k[i] - k[i+1])))
            conv_complex.append(abs((abs(csd(f, x0, h_list[i]) - exact) - abs(csd(f, x0, h_list[i+1]) - exact)) / abs(k[i] - k[i+1])))
    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.plot(k, acc_fd, label = "forward")
    plt.plot(k, acc_cd, label = "centered")
    plt.plot(k, acc_complex, label = "complex step")
    plt.plot(k, acc_auto, label = "algorithmic differentiation")
    plt.legend(fontsize = 10)
    plt.ylabel("accuracy")
    plt.xlabel("k")
    plt.title("Accuracy")

    plt.subplot(3,1,2)
    plt.plot(k, time_fd, label = "forward")
    plt.plot(k, time_cd, label = "centered")
    plt.plot(k, time_complex, label = "complex step")
    plt.plot(k, time_auto, label = "algorithmic differentiation")
    plt.legend(loc = "upper left", fontsize = 8)
    plt.ylabel("secods")
    plt.xlabel("k")
    plt.title("Operation Time")

    plt.subplot(3,1,3)
    plt.plot(k, conv_fd, label = "forward")
    plt.plot(k, conv_cd, label = "centered")
    plt.plot(k, conv_complex, label = "complex step")
    plt.legend(loc = "upper left", fontsize = 8)
    plt.ylabel("rate")
    plt.xlabel("k")
    plt.title("Convergence rate")

    plt.tight_layout()
    plt.show()
