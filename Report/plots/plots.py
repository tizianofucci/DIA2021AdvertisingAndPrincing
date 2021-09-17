import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats 

x = np.linspace(-4, 12, 500)
y = np.heaviside(x, 1) * scipy.stats.norm.pdf(x, 2, 1.5)

x_discr = np.arange(0, 8)
prices = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
for price in prices: 
    distr = scipy.stats.poisson((4.0/(2*((price)/10)+0.5)))
    y1_discr = distr.pmf(x_discr) 
    distr = scipy.stats.poisson((1.0/(2*((price)/10)+0.5)))
    y2_discr = distr.pmf(x_discr) 
    distr = scipy.stats.poisson((3.0/(2*((price)/10)+0.5)))
    y3_discr = distr.pmf(x_discr) 
    plt.xlabel("X")
    plt.ylabel("P(X)")

    plt.vlines(x_discr-0.1, 0, y1_discr, 'r')
    plt.vlines(x_discr, 0, y2_discr, 'g')
    plt.vlines(x_discr+0.1, 0, y3_discr, 'b')
    print("price = {}".format(price))
    plt.legend(["Class 1", "Class 2", "Class 3"])
    plt.show()

