import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__)) 
parentdir = os.path.dirname(currentdir) 
sys.path.append(parentdir)
from ContextEnvironment import ContextEnvironment
from queue import Queue
import numpy as np
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt
from Environment import *
from PricingEnvironment import *
from TS_Learner import *
from ContextGaussianTS_Learner import *
import math
from math import e
from scipy.stats import norm
from UtilFunctions import *
import UtilFunctions

prices = np.array(UtilFunctions.global_prices)
bids = UtilFunctions.global_bids
prod_cost = 3.0
n_arms = 10
bid_idx = 3
bid = bids[bid_idx] 
T = 365
n_experiment = 25
delay = 30

bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
bid_modifiers = [bid_modifiers_c1, bid_modifiers_c2, bid_modifiers_c3]


contexts_prob = np.array([  np.array([np.array([conv_c1(x) for x in prices]),
                            np.array([conv_c1(x) for x in prices])]),
                            np.array([np.array([conv_c2(x) for x in prices]),
                            np.array([conv_c3(x) for x in prices])])])

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]

contexts_mu_base = np.array([ np.array([5,5]) ,
                                np.array([10,10])])


delta_c1 = (compute_delta_customers(bid_modifiers_c1[bid_idx]))
delta_c2 = (compute_delta_customers(bid_modifiers_c2[bid_idx]))
delta_c3 = (compute_delta_customers(bid_modifiers_c3[bid_idx]))


contexts_deltas = np.array([ np.array([delta_c1/2,delta_c1/2]) ,
                                np.array([delta_c2,delta_c3])])

contexts_mu = np.add(contexts_mu_base, contexts_deltas)
print(contexts_mu)

contexts_sigma = np.array([ np.array([math.sqrt(1),math.sqrt(1)]) ,
                         np.array([math.sqrt(1),math.sqrt(1)])])

contexts_bid_offsets = np.array([ np.array([8.0,8.0]) ,
                                 np.array([15.0,5.0])])

contexts_n_returns_coeff = np.array([ np.array([4.0,4.0]) ,
                                 np.array([2.0,3.0])])

"""
Computes expected reward given an arm.

"""
def expected(arm_price,feature_a,feature_b):
    price = prices[arm_price]
    n_returns = (contexts_n_returns_coeff[feature_a][feature_b]/(2*(price/10)+0.5))
    return (contexts_prob[feature_a][feature_b][arm_price]*(price - prod_cost)*(contexts_mu[feature_a][feature_b]) * (n_returns + 1)) - (bid - bid/(contexts_bid_offsets[feature_a][feature_b])) * (contexts_mu[feature_a][feature_b])

opt = []
for i in range(2):
    for j in range(2):
        expected_rewards = [expected(x,i,j) for x in range(len(prices))]
        opt_arm = argmax(expected_rewards)
        opt.append(expected_rewards[opt_arm])
        print("opt_arm: {}, reward: {}".format(opt_arm, expected_rewards[opt_arm]))
opt = np.sum(opt)
print("total optimal reward: {}".format(opt))


ts_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)

"""
Run experiments, collect rewards
"""
for e in range(0,n_experiment):
    env = ContextEnvironment(n_arms,prices, prod_cost, contexts_prob,contexts_bid_offsets,contexts_mu,contexts_sigma,contexts_n_returns_coeff,features_matrix)
    context_gts_learner = ContextGaussianTS_Learner(n_arms,delay,features_matrix)
    pulled_arm_buffer_ts.queue.clear()

    for t in range (0,T):
        pulled_arm_buffer_ts.put(context_gts_learner.pull_arm())

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts,bid)
            context_gts_learner.update(after_30_days_arm_ts,rewards)
            if t>=150 and t%5==0:
                context_gts_learner.try_splitting()

    ts_rewards_per_experiment.append(context_gts_learner.collected_rewards)    
    print(e)

## Plot cumulative regret results
x=np.arange(1,T-delay,1)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r', linewidth = 3)
plt.plot(x, 3800*np.log(x), 'orange', linestyle="dashed", linewidth=3)
plt.legend(["TS", "K log(t)"])
plt.show()

## Plot rewards
x=np.arange(0,T-delay,1)
plt.xlabel("t")
plt.ylabel("Rewards - TS")
plt.plot(x, np.mean(ts_rewards_per_experiment, axis=0),'-ok',color='red', markersize=4, linewidth=0.25)
plt.show()

## Plot Gaussians of each learner, averaged across all experiments
x=np.arange(-100,opt + 100,0.01)
for _ in range(len(context_gts_learner.active_learners)):  
    for i in range(n_arms):
        variance = np.mean(1/context_gts_learner.learners[_].precision_of_rewards[i])
        plt.plot(x, norm.pdf(x, np.mean(context_gts_learner.learners[_].means_of_rewards[i]), math.sqrt(variance)), label="{}".format(i), linewidth = 2)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("P(X)")
    plt.show()


