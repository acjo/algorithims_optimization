# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Caelan Osman
Math 321 Sec 3
September 17, 2020
"""

from matplotlib import pyplot as plt
import numpy as np
import math

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))
    B = A.mean(axis=1)
    variance  = np.var(B)
    return variance

def prob1():
    """This function creates an array of the results of var_of_means()
    with inputs n = 100, 200, ..., 1000. using list comprehension. Then plots and shows
    the resulting array
    """
    results = np.array([var_of_means(n) for n in range(100, 1100, 100)])
    plt.plot(results)
    plt.show()
    return

# Problem 2
def prob2():
    """This function plots sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi].
    """
    x = np.linspace(-2 * np.pi, 2* np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.arctan(x)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.show()
    return

# Problem 3
def prob3():
    """This function plots the function 1/x-1 over [-2,6]"""
    x1 = np.linspace(-2, 0.99, 100)
    x2 = np.linspace(1.01, 6, 100)
    y1 = (x1-1)**-1
    y2 = (x2-1)**-1
    plt.plot(x1, y1, 'm--', linewidth = 4)
    plt.plot(x2, y2, 'm--', linewidth = 4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()
    return

# Problem 4
def prob4():
    """This function plots the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
    """
    x = np.linspace(0, 2 * np.pi, 100)
    #plot of sin(x)
    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x), 'g-')
    ax1.axis([0, 2*np.pi, -2, 2])
    plt.title("Sin(x)", fontsize = 10)
    #plot of sin(2x)
    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2 * x), 'r--')
    ax2.axis([0, 2*np.pi, -2, 2])
    plt.title("Sin(2x)", fontsize = 10)
    #plot of 2sin(x)
    ax3 = plt.subplot(223)
    ax3.plot(x, 2 * np.sin(x), 'b--')
    ax3.axis([0, 2*np.pi, -2, 2])
    ax3.set_title("2sin(x)", fontsize = 10)
    #plot of 2sin(2x)
    ax4 = plt.subplot(224)
    ax4.plot(x, 2 * np.sin(2 * x), 'm:')
    ax4.axis([0, 2*np.pi, -2, 2])
    ax4.set_title("2sin(2x)", fontsize = 10)
    plt.suptitle("Sin Function Variations", fontsize = 14)

    plt.show()
    return

# Problem 5
def prob5():
    """Visualize the data in FARS.npy on a scatter plot and a histogram.
    """
    fars = np.load("FARS.npy")
    dot_subplot = plt.subplot(121)
    plt.plot(fars[:,1], fars[:,2], 'ok', markersize=2)
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.axis("equal")

    box_subplot = plt.subplot(122)
    box_subplot.hist(fars[:,0], bins=np.arange(0,23))
    plt.xlabel("Hours of the Day")
    plt.show()
    return

# Problem 6
def prob6():
    """This function plots the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi] using a heat map and a countour map.
    """

    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = x.copy()
    domainx, domainy = np.meshgrid(x,y)
    image = (np.sin(domainx) * np.sin(domainy))/ (domainx * domainy)

    #heat map
    heat = plt.subplot(121)
    plt.pcolormesh(domainx, domainy, image, cmap='coolwarm')
    plt.colorbar()
    heat.axis([-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])

    #contour map
    cont = plt.subplot(122)
    plt.contour(domainx, domainy, image, 10, cmap='viridis')
    plt.colorbar()
    cont.axis([-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    plt.show()
    return
