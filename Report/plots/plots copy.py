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
plt.plot(x,  y, 'b')
#plt.legend(["TS"])
plt.show()

bids = [0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7]
bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
def delta(modifier): return 200*(modifier*2)

plt.plot(bids, [delta(i) for i in bid_modifiers_c1], 'r')
plt.plot(bids, [delta(i) for i in bid_modifiers_c2], 'g')
plt.plot(bids, [delta(i) for i in bid_modifiers_c3], 'b')
#plt.legend(["TS"])
plt.show()
