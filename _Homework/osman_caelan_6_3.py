import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import norm


#bounds for uniform
a_u = 0
b_u = 1

#bounds for beta
a_b = 1/2.
b_b = 1/2.

n = [1, 2, 4, 8, 16, 32]

x_u = np.linspace(uniform.ppf(a_u), uniform.ppf(b_u), 100)
x_b = np.linspace(beta.ppf(0.01, a_b, b_b), beta.ppf(0.99, a_b, b_b), 100)
x_n = np.linspace(0, 1, 1000)

def mean(n):

    estimates = []
    for i in range(1000):
        current_draw = np.random.normal(0.5, 1/n, n)
        current_estimate = 1/n * (sum(current_draw))
        estimates.append(current_estimate)

    estimates = np.array(estimates)
    return estimates




ax1 = plt.subplot(231)
s_1 = mean(1)
ax1.hist(s_1, density=True, stacked=True, label='x')
ax1.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax1.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax1.plot(x_n, norm.pdf(x_n, 0.5, 1), label = 'normal pdf')
ax1.set_title('n = 1')
plt.ylim(0, 3)
plt.legend(loc='best')

ax2 = plt.subplot(232)
s_2 = mean(2)
ax2.hist(s_2, density=True, stacked=True, label='x')
ax2.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax2.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax2.plot(x_n, norm.pdf(x_n, 0.5, 0.5), label = 'normal pdf')
ax2.set_title('n = 2')
plt.legend(loc='best')

ax3 = plt.subplot(233)
s_3 = mean(4)
ax3.hist(s_3, density=True, stacked=True, label='x')
ax3.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax3.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax3.plot(x_n, norm.pdf(x_n, 0.5, 1/4.), label = 'normal pdf')
ax3.set_title('n = 4')
plt.legend(loc='best')

ax4 = plt.subplot(234)
s_4 = mean(8)
ax4.hist(s_4, density=True, stacked=True, label='x')
ax4.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax4.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax4.plot(x_n, norm.pdf(x_n, 0.5, 1/8.), label = 'normal pdf')
ax4.set_title('n = 8')
plt.legend(loc='best')

ax5 = plt.subplot(235)
s_5 = mean(16)
ax5.hist(s_5, density=True, stacked=True, label='x')
ax5.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax5.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax5.plot(x_n, norm.pdf(x_n, 0.5, 1/16.), label = 'normal pdf')
ax5.set_title('n = 16')
plt.legend(loc='best')

ax6 = plt.subplot(236)
s_6 = mean(32)
ax6.hist(s_6, density=True, stacked=True, label='x')
ax6.plot(x_u, uniform.pdf(x_u), label='uniform pdf')
ax6.plot(x_b, beta.pdf(x_b, a_b, b_b), label = 'beta pdf')
ax6.plot(x_n, norm.pdf(x_n, 0.5, 1/32.), label = 'normal pdf')
ax6.set_title('n = 32')
plt.legend(loc='best')

plt.show()










