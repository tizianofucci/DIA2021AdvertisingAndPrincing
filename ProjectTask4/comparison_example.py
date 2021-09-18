from ContextEnvironment import ContextEnvironment
from queue import Queue
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import seaborn as sns
from Environment import *
from PricingEnvironment import *
from TS_Learner import *
from ContextGaussianTS_Learner import *
import math
from math import e
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
prod_cost = 3.0

n_arms = 10
contexts_prob = np.array([  np.array([np.array([conv_c1(x) for x in prices]),
                            np.array([conv_c1(x) for x in prices])]),
                            np.array([np.array([conv_c2(x) for x in prices]),
                            np.array([conv_c3(x) for x in prices])])])

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
features_column_to_class = [0,0,1,2]

#1.7
bid = 2.5
T = 365
n_experiment = 10
delay = 30
contexts_mu_base = np.array([ np.array([10,10]) ,
                                np.array([10,10])])

#todo fix this
contexts_mu = contexts_mu_base

contexts_sigma = np.array([ np.array([math.sqrt(1),math.sqrt(1)]) ,
                         np.array([math.sqrt(1),math.sqrt(1)])])

contexts_bid_offsets = np.array([ np.array([10,10]) ,
                                 np.array([15,5])])

contexts_n_returns_coeff = np.array([ np.array([4.0,4.0]) ,
                                 np.array([2.0,3.0])])

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

for e in range(0,n_experiment):
    env = ContextEnvironment(n_arms,prices, prod_cost, contexts_prob,contexts_bid_offsets,contexts_mu,contexts_sigma,contexts_n_returns_coeff,features_matrix)
    context_gts_learner = ContextGaussianTS_Learner(n_arms,delay,features_matrix)
    pulled_arm_buffer_ts.queue.clear()

    for t in range (0,T):
        pulled_arm_buffer_ts.put(context_gts_learner.pull_arm())

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards,users_segmentation = env.round(after_30_days_arm_ts,bid)
            context_gts_learner.update(after_30_days_arm_ts,rewards,users_segmentation)
            if t>=130 and t%5==0:
                context_gts_learner.try_splitting()

    ts_rewards_per_experiment.append(context_gts_learner.collected_rewards)    
    print(e)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

x=np.arange(-100,1200,0.01)
for _ in range(len(context_gts_learner.active_learners)):  
    for i in range(n_arms):
        plt.plot(x, norm.pdf(x, context_gts_learner.learners[_].means_of_rewards[i], 1/context_gts_learner.learners[_].precision_of_rewards[i]), label=str(i))
    plt.legend()
    plt.show()


