from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.function_base import linspace
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

bids = [0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7]
bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

bid_modifiers = [bid_modifiers_c1, bid_modifiers_c2, bid_modifiers_c3]

bid_modifiers_aggr = np.zeros(len(bids))
bid_modifiers_aggr = np.sum(bid_modifiers, axis=0)


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
#MAGIC PLOT
#agg = [conv_c1(x)+conv_c2(x)+conv_c3(a) for a in x]
#plt.plot(x,  agg, 'b')
plt.plot(x,  y, 'b', linewidth=3)
plt.xlabel('p')
plt.ylabel('Conversion rate')
plt.show()
aggr = [(150*conv_c1(a)+100*conv_c2(a)+60*conv_c3(a))/310 for a in x]
plt.plot(x, conv_c1(x), 'r', linewidth=3)
plt.plot(x, conv_c2(x), 'g', linewidth=3)
plt.plot(x,  y, 'b', linewidth=3)
plt.plot(x, aggr, 'orange', linestyle="dotted", linewidth=5)
plt.show()
x = bids
y = [200*a for a in bid_modifiers_c1]
y1 = [200*a for a in bid_modifiers_c2]
y2 = [200*a for a in bid_modifiers_c3]
y3 = [200*a for a in bid_modifiers_aggr]
plt.plot(x, y, 'r', linewidth=3)
plt.plot(x, y1, 'g', linewidth=3)
plt.plot(x, y2, 'b', linewidth=3)
plt.plot(x, y3, 'orange', linewidth=3, linestyle="dashed")
plt.legend(["Class 1","Class 2","Class 3","Aggregate"])
plt.xlabel('b')
plt.ylabel('Delta customers')
plt.show()

#plt.show()
