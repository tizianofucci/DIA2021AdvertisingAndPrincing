from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats 
from math import e
x = np.linspace(4.0, 8.5)
#x = np.arange(4.0, 9.0, 0.5)

def conv_c1(x):
    
    return 1.4* e** (-0.14*x)

def conv_c2(x):
    return -0.1 + 5* e** (-0.4*x)
    
def conv_c3(x):
    if x < 6.0:
        return 0.8*e**(-0.5*((x-5.5)**2))
    else:
        return 20 * (e**(-0.557*x))



plt.plot(x, conv_c1(x), 'r', linewidth=3)
plt.xlabel('p')
plt.ylabel('Conversion rate')
plt.show()
plt.plot(x, conv_c2(x), 'g', linewidth=3)
plt.xlabel('p')
plt.ylabel('Conversion rate')
plt.show()
#plt.plot(x, conv_c3(x), 'b')
y = [conv_c3(a) for a in x]
# MAGIC PLOT
# agg = [conv_c1(x)+conv_c2(x)+conv_c3(a) for a in x]
# plt.plot(x,  agg, 'b')
plt.plot(x,  y, 'b', linewidth=3)
plt.xlabel('p')
plt.ylabel('Conversion rate')
plt.show()
aggr = [(150*conv_c1(a)+100*conv_c2(a)+60*conv_c3(a))/310 for a in x]
plt.plot(x, aggr, 'orange', linestyle="dotted", linewidth=5)
plt.xlabel('p')
plt.ylabel('Conversion rate')
plt.legend(["Class 1", "Class 2", "Class 3", "Aggregate"])
plt.show()

#plt.show()
