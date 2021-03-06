#osman_caelan_13_6.py


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fz1 = lambda x, y: -x - (1/2.)*y + (1/2.)

fz2 = lambda x, y: (1/3.) - (1/3.)*x - y


x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)

X, Y = np.meshgrid(x, y)

zeros = np.zeros_like(X)

Z1 = fz1(X, Y)
Z2 = fz2(X, Y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z1)
ax.plot_surface(X, Y, Z2)
'''
ax.plot_surface(X, Y, zeros)
ax.plot_surface(zeros, X, Y)
ax.plot_surface(X, zeros, Y)
'''
plt.show()
