from math import e
import math
import numpy as np

def conv_c1(x):
    
    return 1.4* e** (-0.14*x)

def conv_c2(x):
    return 0.17 + 6* e** (-0.6*x)
    
def conv_c3(x):
    if x < 6.0:
        return 0.8*e**(-0.5*((x-5.5)**2))
    else:
        return 20 * (e**(-0.557*x))

def compute_delta_customers(modifier): 
    return 50*(modifier*2)

global_prices = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
global_bids = [0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7]
