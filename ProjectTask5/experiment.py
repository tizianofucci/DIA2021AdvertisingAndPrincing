import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__)) 
parentdir = os.path.dirname(currentdir) 
sys.path.append(parentdir)
import numpy as np
from numpy.core.fromnumeric import argmax
import math
import matplotlib.pyplot as plt
from queue import Queue
from scipy.stats import norm
from UtilFunctions import *
import UtilFunctions
from BiddingEnvironment import *
from GaussianTS_Learner import *

n_arms = 10
prod_cost = 3.0

T = 365
n_experiment = 100
delay = 30

# Doubled in order to have negative revenue on some arms
bids = 2*np.array(UtilFunctions.global_bids)

bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

bid_modifiers = [bid_modifiers_c1, bid_modifiers_c2, bid_modifiers_c3]

bid_modifiers_aggr = np.zeros(len(bids))
bid_modifiers_aggr = np.sum(bid_modifiers, axis=0)
print(bid_modifiers_aggr)

mu_new_customer = 10
sigma_new_customer = math.sqrt(1)


prices = UtilFunctions.global_prices
class_probs = np.array([[conv_c1(x) for x in prices],[conv_c2(x) for x in prices],[conv_c3(x) for x in prices]])
price_idx = 4
price = prices[price_idx]

deltas = np.array([])

for i in range(len(bid_modifiers)):
    temp_deltas = np.array([])
    for j in range(len(bids)):
        temp_deltas = np.append(temp_deltas,compute_delta_customers(bid_modifiers[i][j]) + mu_new_customer)
    deltas = np.append(deltas,temp_deltas)
deltas = deltas.reshape((3,10))
coeffs = [4.0, 2.0, 3.0]
avg_coeffs = []
for i in range(len(bids)):
    avg_coeffs.append(np.average(coeffs, weights = deltas[:,i]))

bid_offsets = [8.0, 5.0, 15.0]
avg_bid_offsets = []
for i in range(len(bids)):
    avg_bid_offsets.append(np.average(bid_offsets, weights = deltas[:,i])) 

conv_rates = []
for i in range(len(bids)):
    conv_rates.append(np.average(np.array(class_probs[:,price_idx]), weights = deltas[:,i]))

"""
Computes expected reward given an arm.

"""
def expected(arm):
    bid = bids[arm]
    expected_returns = (avg_coeffs[arm]/(2*((price)/10)+0.5))
    delta_customers = compute_delta_customers(bid_modifiers_aggr[arm])
    n_visits = mu_new_customer*3 + delta_customers
    conv_rate = conv_rates[arm]
    return (conv_rate*n_visits*(price - prod_cost)*(expected_returns + 1)) - (bid - bid/avg_bid_offsets[arm]) * n_visits

expected_rewards = [expected(x) for x in range(n_arms)]
print("expected rewards:\n", expected_rewards)

opt_arm = argmax(expected_rewards)
opt = expected_rewards[opt_arm]

print("optimal reward: {}; with arm {}".format(opt, opt_arm))


ts_rewards_per_experiment = []

gts_means_of_rewards = []
gts_precision_of_rewards = []


pulled_arm_buffer_ts = Queue(maxsize=31)
pulled_arm_buffer_ucb = Queue(maxsize=31)

opt_count = 0

"""
Run experiments, collect rewards
"""
for e in range(0,n_experiment):
    env = BiddingEnvironment(n_arms, price, prod_cost, bids, bid_modifiers_aggr, conv_rates, mu_new_customer*3, sigma_new_customer,avg_coeffs,avg_bid_offsets)
    gts_learner = GaussianTS_Learner(n_arms,delay)

    pulled_arm_buffer_ts.queue.clear()
    

    for t in range (0,T):
        
        try:
            pulled_arm_buffer_ts.put(gts_learner.pull_arm())
        
        except Exception as err:
            print("Expected negative revenue on all arms")
            break

        if t>=delay:

            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts)
            gts_learner.update(after_30_days_arm_ts,rewards)

    ts_rewards_per_experiment.append(gts_learner.collected_rewards)
    gts_means_of_rewards.append(gts_learner.means_of_rewards)
    gts_precision_of_rewards.append(gts_learner.precision_of_rewards)

    print(e)
    n_pulls_per_arm = [len(x) for x in gts_learner.rewards_per_arm]
    print(n_pulls_per_arm)
    if (argmax(n_pulls_per_arm) == opt_arm):
        opt_count +=1

## Plot of cumulative regret   
x=np.arange(1,T-delay,1)    
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r', linewidth = 3)
plt.plot(x, 2500*np.log(x), 'orange', linestyle="dashed", linewidth=3)
plt.legend(["TS", "K log(t)"])
plt.show()
gts_means_of_rewards = np.transpose(gts_means_of_rewards)
gts_precision_of_rewards = np.transpose(gts_precision_of_rewards)

## Plot daily rewards

x=np.arange(0,T-delay,1)
plt.xlabel("t")
plt.ylabel("Rewards - TS")
plt.plot(x, np.mean(ts_rewards_per_experiment, axis=0),'-ok',color='red', markersize=4, linewidth=0.25)
plt.show()

## Plot average of Gaussians per arm
x=np.arange(-100,opt*1.2,0.01)
for i in range(n_arms):
    variance = np.mean(1/gts_precision_of_rewards[i])
    plt.plot(x, norm.pdf(x, np.mean(gts_means_of_rewards[i]), math.sqrt(variance)), label="{}".format(i), linewidth = 2)
plt.legend()
plt.xlabel("X")
plt.ylabel("P(X)")
plt.show()