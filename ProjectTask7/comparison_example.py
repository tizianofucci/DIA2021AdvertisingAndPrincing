from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from ContextEnvironment import ContextEnvironment
from queue import Queue
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import seaborn as sns
from Environment import *
from PricingBiddingEnvironment import *
from ContextGPTS_Learner import *
import math
from math import e
from matplotlib import cm
from scipy.stats import norm


def conv_c1(x):
    
    return 1.4* e** (-0.14*x)

def conv_c2(x):
    return 0.1 + 6* e** (-0.6*x)
    
def conv_c3(x):
    if x < 6.0:
        return 0.8*e**(-0.5*((x-5.5)**2))
    else:
        return 20 * (e**(-0.557*x))


prices = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
n_arms = 10
contexts_prob = np.array([  np.array([np.array([conv_c1(x) for x in prices]),
                            np.array([conv_c1(x) for x in prices])]),
                            np.array([np.array([conv_c2(x) for x in prices]),
                            np.array([conv_c3(x) for x in prices])])])

prod_cost = 3.0

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
features_column_to_class = [0,0,1,2]

bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])

bid_modifiers = np.array([  np.array([np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]),
                            np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4])]),
                            np.array([np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])])

T = 200
n_experiment = 1
delay = 30
contexts_mu = np.array([ np.array([10,10]) ,
                         np.array([10,10])])
contexts_sigma = np.array([ np.array([math.sqrt(4),math.sqrt(4)]) ,
                         np.array([math.sqrt(6),math.sqrt(8)])])

contexts_bid_offsets = np.array([ np.array([10,10]) ,
                                 np.array([15,5])])

contexts_n_returns_coeffs = np.array([ np.array([4.0,4.0]) ,
                                 np.array([2.0,3.0])])


def expected(arm_bids,arm_price,feature_a,feature_b):
    bid = bids[arm_bids]
    price = prices[arm_price]
    n_returns = (contexts_n_returns_coeffs[feature_a][feature_b]/(2*(price/10)+0.5))
    bid_offset = contexts_bid_offsets[feature_a][feature_b]
    delta_customers = 50*(bid_modifiers[feature_a][feature_b][arm_bids]*2)
    return (contexts_prob[feature_a][feature_b][arm_price]*(price - prod_cost)*(contexts_mu[feature_a][feature_b] +delta_customers) * (n_returns + 1)) - (bid - bid/bid_offset) * (contexts_mu[feature_a][feature_b] + delta_customers)

opt = []
for i in range(2):
    for j in range(2):
        expected_rewards = [expected(x,y,i,j) for x in range(len(bids)) for y in range(len(prices))]
        opt_arm = argmax(expected_rewards)
        print(np.unravel_index(opt_arm,(10,10)))
        opt.append(expected_rewards[opt_arm])
        print(expected_rewards[opt_arm])
opt = np.sum(opt)
print(opt)

ts_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)

for e in range(0,n_experiment):
    env = ContextEnvironment(prices,prod_cost,bids,bid_modifiers,contexts_bid_offsets, contexts_prob,contexts_mu,contexts_sigma, contexts_n_returns_coeffs, features_matrix)
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
            if t>=160 and t%5==0:
                context_gpts_learner.try_splitting()
        if t%20 ==0:
            print(t)
    ts_rewards_per_experiment.append(context_gpts_learner.collected_rewards)    
    print(e)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

X, Y =  np.meshgrid(prices,bids)
for i in range(len(context_gpts_learner.learners)):
    if context_gpts_learner.active_learners[i] == True:
        Z = context_gpts_learner.learners[i].means.reshape(len(bids),len(prices))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()