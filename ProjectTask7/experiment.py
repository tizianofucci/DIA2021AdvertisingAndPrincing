from sklearn.exceptions import ConvergenceWarning
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import math
import matplotlib.pyplot as plt
from queue import Queue
from scipy.stats import norm

from PricingBiddingEnvironment import *
from GPTS_Learner import *

n_arms = 11

prices = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
prod_cost = 3.0

first_buy_probabilities = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1])
T = 230
n_experiment = 3
delay = 30
mu_new_customer = 12
sigma_new_customer = math.sqrt(1)


#bids = np.array([0.5,0.7,0.9,1.1,1.4,1.5,1.7,1.9,2.1,2.3])
#bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])
bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])

#bid_modifiers = np.array([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,1.8])
bid_modifiers = np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4])



def expected(arm_bids,arm_price):
    bid = bids[arm_bids]
    price = prices[arm_price]
    delta_customers = 200*(bid_modifiers[arm_bids]*2)
    return first_buy_probabilities[arm_price]*(mu_new_customer + delta_customers)*(price - prod_cost)*((3.0/(2*((price)/10)+0.5)) + 1) - (bid - bid/10) * (mu_new_customer + delta_customers)

expected_rewards = [expected(x,y) for x in range(len(bids)) for y in range(len(prices))]
print("expected rewards:\n", expected_rewards)

opt_arm = argmax(expected_rewards)
opt = expected_rewards[opt_arm]

print(np.unravel_index(opt_arm,(len(bids),len(prices))))


ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []


pulled_arm_buffer_ts = Queue(maxsize=31)
opt_count = 0

for e in range(0,n_experiment):
    env = PricingBiddingEnvironment(prices, prod_cost, bids, bid_modifiers, first_buy_probabilities, mu_new_customer, sigma_new_customer)
    gpts_learner = GPTS_Learner(len(bids),len(prices),[bids,prices],delay)

    pulled_arm_buffer_ts.queue.clear()
    

    for t in range (0,T):
        
        # try:
        pulled_arm_buffer_ts.put(gpts_learner.pull_arm())
        
        # except Exception as err:
        #     print("Expected negative revenue on all arms")
        #     break

        if t>=delay:

            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts)
            gpts_learner.update(after_30_days_arm_ts,rewards)
#        print(t)

    ts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    #print(env.results)
    print(e)
    n_pulls_per_arm = [len(x) for x in gpts_learner.rewards_per_arm]
    print(n_pulls_per_arm)
    if (argmax(n_pulls_per_arm) == opt_arm):
        opt_count +=1
    

print("optimal arm found {} times out of {}".format(opt_count ,n_experiment))
#print("optimal= {}".format(opt))
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

X, Y =  np.meshgrid(prices,bids)
Z = gpts_learner.means.reshape(len(bids),len(prices))



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# x=np.arange(-100,600,0.01)
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[0], 1/gts_learner.precision_of_rewards[0]), label='0')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[1], 1/gts_learner.precision_of_rewards[1]), label='1')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[2], 1/gts_learner.precision_of_rewards[2]), label='2')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[3], 1/gts_learner.precision_of_rewards[3]), label='3')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[4], 1/gts_learner.precision_of_rewards[4]), label='4')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[5], 1/gts_learner.precision_of_rewards[5]), label='5')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[6], 1/gts_learner.precision_of_rewards[6]), label='6')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[7], 1/gts_learner.precision_of_rewards[7]), label='7')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[8], 1/gts_learner.precision_of_rewards[8]), label='8')
# plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[9], 1/gts_learner.precision_of_rewards[9]), label='9')

# #for i in range(len(gts_learner.means_of_rewards)):
#     #print("mean: {}, cdf in 0: {}".format(gts_learner.means_of_rewards[i],norm.cdf(0, gts_learner.means_of_rewards[i], 1/gts_learner.precision_of_rewards[i])))

# plt.legend()
# plt.show()
