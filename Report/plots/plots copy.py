import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats 
from math import e
#2* e** (-0.2*x)
def conv_c1(x):
    return 1.4* e** (-0.1*x)

def conv_c2(x):
    return 0.1 + 6* e** (-0.6*x)
    
def conv_c3(x):
    return 4.5* e** (-0.45*x)

def conv_c5(x):
    if x < 6.0:
        return 0.8*e**(-0.5*((x-5.5)**2))
    else:
        return 20 * (e**(-0.557*x))


x = np.linspace(4.0, 9.0)
#x = np.arange(4.0, 9.0, 0.5)


plt.xlabel = ("Price")
plt.ylabel = ("Conversion rate")
plt.plot(x, conv_c1(x), 'r')
plt.plot(x, conv_c2(x), 'g')
#plt.plot(x, conv_c3(x), 'b')
y = [conv_c5(a) for a in x]
plt.plot(x,  y, 'purple')
#plt.legend(["TS"])
plt.show()