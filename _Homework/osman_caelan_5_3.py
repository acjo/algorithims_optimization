#osman_caelan_5_3.py
'''
Caelan osman
Math 320 Sec 1
October 25, 2020
'''

import numpy as np
from matplotlib import pyplot as plt

def prob5_16():
    #graph as a function of false positive
    P_d = 0.0004 #probability of having the disease
    P_T_pos_H = np.linspace(0.001, 0.1, 100) #probabilities of having a false positive (step size 0.001)
    P_T_neg_D = 0.05 #probability of having a false negative

    #use this function to plot the probability of having the disease as a function of probabilities of false positives
    D_T_pos_1 = lambda x: ((1 - P_T_neg_D) * P_d) /(((1 - P_T_neg_D) * P_d) + (x * (1 - P_d)))

    #first subplot
    ax1 = plt.subplot(131)
    ax1.plot(P_T_pos_H, D_T_pos_1(P_T_pos_H))
    ax1.set_title('Probability of Having the Disease vs False Positives')
    plt.xlabel('False Positives')
    plt.ylabel('Having the Disease')

    #graph as a function of false negative
    P_T_pos_H = 0.05 #set the single value of the false positive
    P_T_neg_D = np.linspace(0.001, 0.1, 100) #range probabilities of having a false negative

    #use this function to plot the probability of having the disease as a function of probabilities of false negatives
    D_T_pos_2 = lambda x: ((1-x) * P_d) / (((1-x) * P_d ) + (P_T_pos_H * (1 - P_d)))

    #second subplot
    ax2 = plt.subplot(132)
    ax2.plot(P_T_neg_D, D_T_pos_2(P_T_neg_D))
    ax2.set_title('Probability of Having the Disease vs False Negatives')
    plt.xlabel('False Negatives')
    plt.ylabel('Having the Disease')

    #graph as a function of incidence
    P_T_neg_D = 0.05 #reset false negative rate
    P_d = np.linspace(0.001, 0.05, 50) #vary incidence rate

    #use this function to plot the probability of having the disease as a function of probabilities of incidences
    D_T_pos_3 = lambda x: ((1 - P_T_neg_D) * x)/ (((1 - P_T_neg_D) * x) + (P_T_pos_H * (1 - x)))

    ax3 = plt.subplot(133)
    ax3.plot(P_d, D_T_pos_3(P_d))
    ax3.set_title('Probability of Having the Disease vs Incidences')
    plt.xlabel('Incidences')
    plt.ylabel('Having the Disease')
    plt.show()

