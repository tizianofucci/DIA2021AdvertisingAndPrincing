import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__)) 
parentdir = os.path.dirname(currentdir) 
sys.path.append(parentdir)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from ContextEnvironment import ContextEnvironment
from queue import Queue
import numpy as np
from numpy.core.fromnumeric import argmax, mean
import matplotlib.pyplot as plt
from Environment import *
from PricingBiddingEnvironment import *
from ContextGPTS_Learner import *
import math
from math import e
from matplotlib import cm
from UtilFunctions import *
import UtilFunctions

vector_of_Z = [[] for i in range(9)]
prices = np.array(UtilFunctions.global_prices)
bids = UtilFunctions.global_bids
prod_cost = 3.0
n_arms = 10
T = 300
n_experiment = 1
delay = 30
contexts_prob = np.array([  np.array([np.array([conv_c1(x) for x in prices]),
                            np.array([conv_c1(x) for x in prices])]),
                            np.array([np.array([conv_c2(x) for x in prices]),
                            np.array([conv_c3(x) for x in prices])])])

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
features_column_to_class = [0,0,1,2]


bid_modifiers = np.array([  np.array([np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]),
                            np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4])]),
                            np.array([np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                            np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])])])


contexts_mu = np.array([ np.array([5,5]) ,
                         np.array([10,10])])
contexts_sigma = np.array([ np.array([math.sqrt(1),math.sqrt(1)]) ,
                         np.array([math.sqrt(1),math.sqrt(1)])])

contexts_bid_offsets = np.array([ np.array([8.0,8.0]) ,
                                 np.array([15.0,5.0])])

contexts_n_returns_coeffs = np.array([ np.array([4.0,4.0]) ,
                                 np.array([2.0,3.0])])

#Needed because 00 and 01 customers belong to same class
delta_customers_multipliers = np.array([ np.array([0.5,0.5]) ,
                                        np.array([1.0,1.0])])
"""
Computes expected reward given an arm.

"""
def expected(arm_bids,arm_price,feature_a,feature_b):
    bid = bids[arm_bids]
    price = prices[arm_price]
    n_returns = (contexts_n_returns_coeffs[feature_a][feature_b]/(2*(price/10)+0.5))
    bid_offset = contexts_bid_offsets[feature_a][feature_b]
    delta_customers = 200*(bid_modifiers[feature_a][feature_b][arm_bids])*(delta_customers_multipliers[feature_a][feature_b])
    return (contexts_prob[feature_a][feature_b][arm_price]*(price - prod_cost)*(contexts_mu[feature_a][feature_b] +delta_customers) * (n_returns + 1)) - (bid - bid/bid_offset) * (contexts_mu[feature_a][feature_b] + delta_customers)

opt_arms = []
opt_rewards = []
for i in range(2):
    for j in range(2):
        expected_rewards = [expected(x,y,i,j) for x in range(len(bids)) for y in range(len(prices))]
        opt_arm = argmax(expected_rewards)
        opt_rewards.append(expected_rewards[opt_arm])
        opt_arms.append(np.unravel_index(opt_arm,(10,10)))
opt = np.sum(opt_rewards)

print("optimal rewards:{}, sum:{}, with arms:{}".format(opt_rewards,opt, opt_arms))

ts_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)

"""
Run experiments, collect rewards
"""
for e in range(0,n_experiment):
    env = ContextEnvironment(prices,prod_cost,bids,bid_modifiers,contexts_bid_offsets, contexts_prob,contexts_mu,contexts_sigma, delta_customers_multipliers, contexts_n_returns_coeffs, features_matrix)
    context_gpts_learner = ContextGPTS_Learner(len(bids),len(prices),[bids,prices],delay,features_matrix)
    pulled_arm_buffer_ts.queue.clear()

    for t in range (0,T):


        try:
            pulled_arm_buffer_ts.put(context_gpts_learner.pull_arm())
        
        except Exception as err:
             print("Expected negative revenue on all arms")
             break

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards,users_segmentation = env.round(after_30_days_arm_ts)
            context_gpts_learner.update(after_30_days_arm_ts,rewards,users_segmentation)
            if t>=140 and t%5==0:
                context_gpts_learner.try_splitting()
        # if t%20 ==0:
        #     print(t)
    ts_rewards_per_experiment.append(context_gpts_learner.collected_rewards)
    for i in range(9):
        Z_e = context_gpts_learner.learners[i].means.reshape(len(bids),len(prices))
        vector_of_Z[i].append(Z_e)
    print(e)

## Plot regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

## Plot daily rewards

x=np.arange(0,T-delay,1)
plt.xlabel("t")
plt.ylabel("Rewards - TS")
plt.plot(x, np.mean(ts_rewards_per_experiment, axis=0),'-ok',color='red', markersize=4, linewidth=0.25)
plt.show()

##Plot average means of GPTS

X, Y =  np.meshgrid(prices,bids)
for i in range(len(context_gpts_learner.learners)):
    Z = mean(vector_of_Z[i],axis=0)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Price")
    ax.set_ylabel("Bid")
    ax.set_zlabel("Reward")
plt.show()

