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
from scipy.stats import norm

n_arms = 10
contexts_prob = np.array([  np.array([np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]),
                            np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])]),
                            np.array([np.array([0.6, 0.55, 0.5, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.0]),
                            np.array([0.3, 0.4, 0.7, 0.7, 0.6, 0.6, 0.55, 0.55, 0.5, 0.5])])])
prices = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
prod_cost = 3.0

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
features_column_to_class = [0,0,1,2]

bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])

bid_modifiers = np.array([  np.array([np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]),
                            np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4])]),
                            np.array([np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])])])

T = 115
n_experiment = 5
delay = 30
contexts_mu = np.array([ np.array([10,10]) ,
                         np.array([10,10])])
contexts_sigma = np.array([ np.array([math.sqrt(4),math.sqrt(4)]) ,
                         np.array([math.sqrt(6),math.sqrt(8)])])

# opt = (contexts_prob[0][1][0]*prices[0]*contexts_mu[0][1] * (3.0/(2*(prices[0] - 3.5))+1)) - (bid - bid/10) * contexts_mu[0][1] + \
#      (contexts_prob[0][1][0]*prices[0]*contexts_mu[0][1] * (3.0/(2*(prices[0] - 3.5))+1)) - (bid - bid/10) * contexts_mu[0][1] + \
#     (contexts_prob[1][0][9]*prices[9]*contexts_mu[1][0] * (3.0/(2*(prices[9] - 3.5))+1)) - (bid - bid/10) * contexts_mu[1][0] + \
#     (contexts_prob[1][1][2]*prices[2]*contexts_mu[1][1] * (3.0/(2*(prices[2] - 3.5))+1)) - (bid - bid/10) * contexts_mu[1][1]


def expected(arm_bids,arm_price,feature_a,feature_b):
    bid = bids[arm_bids]
    price = prices[arm_price]
    delta_customers = 200*(bid_modifiers[feature_a][feature_b][arm_bids]*2)
    return (contexts_prob[feature_a][feature_b][arm_price]*(price - prod_cost)*(contexts_mu[feature_a][feature_b] +delta_customers) * ((3.0/(2*((price)/10)+0.5)) + 1)) - (bid - bid/10) * (contexts_mu[feature_a][feature_b] + delta_customers)

opt = []
for i in range(2):
    for j in range(2):
        expected_rewards = [expected(x,y,i,j) for x in range(len(bids)) for y in range(len(prices))]
        #print("expected rewards:\n", expected_rewards)
        opt_arm = argmax(expected_rewards)
        print(np.unravel_index(opt_arm,(10,10)))
        opt.append(expected_rewards[opt_arm])
        print(expected_rewards[opt_arm])
opt = np.sum(opt)
print(opt)


#mu = 0
#variance = 1
#sigma = math.sqrt(variance)
#x = np.linspace(mu_new_customer - 3*sigma_new_customer, mu_new_customer + 3*sigma_new_customer, 100)
#plt.plot(x, stats.norm.pdf(x, mu_new_customer, sigma_new_customer))
#plt.show()

ts_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)

for e in range(0,n_experiment):
    env = ContextEnvironment(prices,prod_cost,bids,bid_modifiers,contexts_prob,contexts_mu,contexts_sigma,features_matrix)
    context_gpts_learner = ContextGPTS_Learner(len(bids),len(prices),[bids,prices],delay,features_matrix)
    pulled_arm_buffer_ts.queue.clear()

    for t in range (0,T):
        pulled_arm_buffer_ts.put(context_gpts_learner.pull_arm())

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards,users_segmentation = env.round(after_30_days_arm_ts)
            context_gpts_learner.update(after_30_days_arm_ts,rewards,users_segmentation)
            if t>=80 and t%5==0:
                context_gpts_learner.try_splitting()
        if t%20 ==0:
            print(t)
    ts_rewards_per_experiment.append(context_gpts_learner.collected_rewards)    
    print(e)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS","UCB"])
plt.show()

# x=np.arange(50,700,0.01)
# for _ in range(len(context_gpts_learner.active_learners)):  
#     for i in range(n_arms):
#         plt.plot(x, norm.pdf(x, context_gpts_learner.learners[_].means_of_rewards[i], 1/context_gpts_learner.learners[_].precision_of_rewards[i]), label=str(i))
#     plt.legend()
#     plt.show()


