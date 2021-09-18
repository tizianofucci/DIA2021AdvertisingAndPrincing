import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__)) 
parentdir = os.path.dirname(currentdir) 
sys.path.append(parentdir)
from sklearn.exceptions import ConvergenceWarning
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import math
from math import e
import matplotlib.pyplot as plt
from queue import Queue
from PricingBiddingEnvironment import *
from GPTS_Learner import *
from UtilFunctions import *
import UtilFunctions

n_arms = 10
T = 230
n_experiment = 3
delay = 30
mu_new_customer = 10
sigma_new_customer = math.sqrt(1)
prod_cost = 3.0

bids = UtilFunctions.global_bids

bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

bid_modifiers = [bid_modifiers_c1, bid_modifiers_c2, bid_modifiers_c3]

bid_modifiers_aggr = np.zeros(len(bids))
bid_modifiers_aggr = np.sum(bid_modifiers, axis=0)

prices = UtilFunctions.global_prices
class_probs = [[conv_c1(x) for x in prices],[conv_c2(x) for x in prices],[conv_c3(x) for x in prices]]

visits_per_class = np.array([])

for i in range(len(bid_modifiers)):
    temp_deltas = np.array([])
    for j in range(len(bids)):
        temp_deltas = np.append(temp_deltas,compute_delta_customers(bid_modifiers[i][j]) + mu_new_customer)
    visits_per_class = np.append(visits_per_class, temp_deltas)
visits_per_class = visits_per_class.reshape((3,10))



coeffs = [4.0, 2.0, 3.0]
avg_coeffs = []
for i in range(len(bids)):
    avg_coeffs.append(np.average(coeffs, weights = visits_per_class[:,i]))

bid_offsets = [8.0, 5.0, 15.0]
avg_bid_offsets = []
for i in range(len(bids)):
    avg_bid_offsets.append(np.average(bid_offsets, weights = visits_per_class[:,i])) 


prob_by_bid = [np.average(class_probs, axis=0, weights = visits_per_class[:,i]) for i in range(len(bids))]


def expected(arm_bids,arm_price):
    bid = bids[arm_bids]
    price = prices[arm_price]
    expected_returns = (avg_coeffs[arm_bids]/(2*((price)/10)+0.5))
    delta_customers = compute_delta_customers(bid_modifiers_aggr[arm_bids])
    n_visits = mu_new_customer*3+delta_customers
    bid_offset = avg_bid_offsets[arm_bids]
    conv_rate = prob_by_bid[arm_bids][arm_price]
    return conv_rate*n_visits*(price - prod_cost)*(expected_returns + 1) - (bid - bid/bid_offset) * (n_visits)

expected_rewards = [expected(x,y) for x in range(len(bids)) for y in range(len(prices))]
# print("expected rewards:\n", expected_rewards)

opt_arm = argmax(expected_rewards)
opt = expected_rewards[opt_arm]
opt_tuple = np.unravel_index(opt_arm,(len(bids),len(prices)))
print("optimal reward: {}; with arm {}".format(opt, opt_tuple))

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []


pulled_arm_buffer_ts = Queue(maxsize=31)
opt_count = 0

for e in range(0,n_experiment):
    env = PricingBiddingEnvironment(prices, prod_cost, bids, bid_modifiers_aggr, prob_by_bid, mu_new_customer*3, sigma_new_customer, returns_coeffs=avg_coeffs, bid_offsets=avg_bid_offsets)
    gpts_learner = GPTS_Learner(len(bids),len(prices),[bids,prices],delay)

    pulled_arm_buffer_ts.queue.clear()
    

    for t in range (0,T):
        
        try:
            pulled_arm_buffer_ts.put(gpts_learner.pull_arm())
        
        except Exception as err:
             print("Expected negative revenue on all arms")
             break

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts)
            gpts_learner.update(after_30_days_arm_ts,rewards)
            
        if t%20 == 0: print(t)

    ts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    print(e)


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
