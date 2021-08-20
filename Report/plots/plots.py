import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats 

x = np.linspace(-4, 12, 500)
y = np.heaviside(x, 1) * scipy.stats.norm.pdf(x, 2, 1.5)

x_discr = np.arange(0, 8)
prices = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
for price in prices: 
    distr = scipy.stats.poisson(3.0/(2*(price-3.5)))
    y_discr = distr.pmf(x_discr) 
    #y = np.max(0, scipy.stats.norm.pdf(x, 3, 3))
    #plt.figure(0)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.vlines(x_discr, 0, y_discr, 'r')
    print("price = {}".format(price))
    plt.show()
    #print(z)