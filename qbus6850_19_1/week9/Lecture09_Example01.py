# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:56:31 2017

@author: cwan6954
revised by Professor Junbin Gao, 8 August 2018
"""
import numpy as np
import matplotlib.pyplot as plt
#############################
z= 0.5
1/(1+np.exp(-z))

# generate plot of sigmoid Function
x_list= np.linspace(-6,6,100)
# s impact
y_list_1= 1/(1+np.exp(-1*(x_list-0)))
y_list_2= 1/(1+np.exp(-2*(x_list-0)))
y_list_3= 1/(1+np.exp(-3*(x_list-0)))

plt.plot(x_list, y_list_1, label = "s=1,l=0")
plt.plot(x_list, y_list_2, label = "s=2,l=0")
plt.plot(x_list, y_list_3, label = "s=3,l=0")
plt.legend(loc="upper left")
# l impact
x_list= np.linspace(-6,8,100)
y_list_1= 1/(1+np.exp(-1*(x_list-0)))
y_list_2= 1/(1+np.exp(-1*(x_list-1)))
y_list_3= 1/(1+np.exp(-1*(x_list-2)))

plt.plot(x_list, y_list_1, label = "s=1,l=0")
plt.plot(x_list, y_list_2, label = "s=1,l=1")
plt.plot(x_list, y_list_3, label = "s=1,l=2")
plt.legend(loc="upper left")
#################33
y_list_1= 1/(1+np.exp(-1*(x_list-0)))
y_list_3= 1/(1+np.exp(-3*(x_list-0)))

plt.plot(x_list, y_list_1, label = "s=1,l=0", linestyle= "dotted")
plt.plot(x_list, y_list_3, label = "s=3,l=0")

plt.legend(loc="upper left")

########################
# generate plot of Tangent and Hyperbolic Tangent Function
x_list= np.linspace(-6,6,100)
y_list_1= 2/(1+np.exp(-2*x_list))-1
y_list_2= (np.exp(x_list)-np.exp(-x_list))/(np.exp(x_list)+np.exp(-x_list))
plt.plot(x_list, y_list_1, label = "Tangent function")
plt.plot(x_list, y_list_2, label = "Hyperbolic Tangent function")
plt.legend(loc="upper left")   

######################################################
# def sigmoid_prime(self,z):
# derivative of the Sigmoid fucntion
x_list= np.linspace(-6,6,100)
y_list= 1/(1+np.exp(-x_list))
y_list_1= np.exp(-x_list)/((1+np.exp(-x_list))**2)
plt.plot(x_list, y_list, label = "Sigmoid function")
plt.plot(x_list, y_list_1, label = "Sigmoid function derivative")
plt.legend(loc="upper left")   
#############################








































    