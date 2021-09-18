import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats 

x = np.linspace(-4, 12, 500)
y = np.heaviside(x, 1) * scipy.stats.norm.pdf(x, 2, 1.5)

coeff_c1 = 4.0
coeff_c2 = 2.0
coeff_c3 = 3.0

coeffs = [coeff_c1, coeff_c2, coeff_c3]

x_discr = np.arange(0, 8)
prices = [4.0, 5.5, 7.0, 8.5]
for price in prices:
    distr = scipy.stats.poisson((coeff_c1/(2*((price)/10)+0.5)))
    y1_discr = distr.pmf(x_discr) 
    distr = scipy.stats.poisson((coeff_c2/(2*((price)/10)+0.5)))
    y2_discr = distr.pmf(x_discr) 
    distr = scipy.stats.poisson((coeff_c3/(2*((price)/10)+0.5)))
    y3_discr = distr.pmf(x_discr) 
    aggr = [(150*y1_discr+100*y2_discr+60*y3_discr)/310]

    means = [(coeff/(2*((price)/10)+0.5)) for coeff in coeffs]
    distr = scipy.stats.poisson(np.mean(means))
    y_mean_discr = distr.pmf(x_discr) 

    distr = scipy.stats.poisson((np.mean(coeffs)/(2*((price)/10)+0.5)))
    y_meanc_discr = distr.pmf(x_discr) 

    plt.xlabel("X")
    plt.ylabel("P(X)")

    plt.vlines(x_discr-0.3, 0, y1_discr, 'r', linewidth=5)
    plt.vlines(x_discr-0.1, 0, y2_discr, 'g', linewidth=5)
    plt.vlines(x_discr+0.1, 0, y3_discr, 'b', linewidth=5)
    plt.vlines(x_discr+0.3, 0, aggr, 'orange', linestyle="dashed", linewidth=5)
    #plt.vlines(x_discr+0.2, 0, y_mean_discr, 'purple')
    #plt.vlines(x_discr+0, 0, y_meanc_discr, 'purple')
    print("price = {}".format(price))
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregate"])
    plt.show()

