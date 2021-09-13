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
from scipy.stats import norm

n_arms = 10
contexts_prob = np.array([  np.array([np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]),
                            np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])]),
                            np.array([np.array([0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                            np.array([0.3, 0.4, 0.7, 0.7, 0.6, 0.6, 0.55, 0.55, 0.5, 0.5])])])
prices = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])

features_matrix = [[0,0],
                   [0,1],
                   [1,0],
                   [1,1]]
features_column_to_class = [0,0,1,2]

bid = 0.2
T = 1000
n_experiment = 30
delay = 30
contexts_mu = np.array([ np.array([10,10]) ,
                         np.array([20,50])])
contexts_sigma = np.array([ np.array([math.sqrt(4),math.sqrt(4)]) ,
                         np.array([math.sqrt(6),math.sqrt(8)])])

opt = (contexts_prob[0][1][0]*prices[0]*contexts_mu[0][1] * (3.0/(2*(prices[0] - 3.5))+1)) - (bid - bid/10) * contexts_mu[0][1] + \
     (contexts_prob[0][1][0]*prices[0]*contexts_mu[0][1] * (3.0/(2*(prices[0] - 3.5))+1)) - (bid - bid/10) * contexts_mu[0][1] + \
    (contexts_prob[1][0][9]*prices[9]*contexts_mu[1][0] * (3.0/(2*(prices[9] - 3.5))+1)) - (bid - bid/10) * contexts_mu[1][0] + \
    (contexts_prob[1][1][2]*prices[2]*contexts_mu[1][1] * (3.0/(2*(prices[2] - 3.5))+1)) - (bid - bid/10) * contexts_mu[1][1]

#mu = 0
#variance = 1
#sigma = math.sqrt(variance)
#x = np.linspace(mu_new_customer - 3*sigma_new_customer, mu_new_customer + 3*sigma_new_customer, 100)
#plt.plot(x, stats.norm.pdf(x, mu_new_customer, sigma_new_customer))
#plt.show()

ts_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)

for e in range(0,n_experiment):
    env = ContextEnvironment(n_arms,prices,contexts_prob,contexts_mu,contexts_sigma,features_matrix)
#    env = PricingEnvironment(n_arms,prices,contexts_prob,contexts_mu,contexts_sigma,features_matrix)
    context_gts_learner = ContextGaussianTS_Learner(n_arms,delay,features_matrix)
    pulled_arm_buffer_ts.queue.clear()

    for t in range (0,T):
        pulled_arm_buffer_ts.put(context_gts_learner.pull_arm())

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards,users_segmentation = env.round(after_30_days_arm_ts,bid)
            context_gts_learner.update(after_30_days_arm_ts,rewards,users_segmentation)
            context_gts_learner.try_splitting(users_segmentation)

    ts_rewards_per_experiment.append(context_gts_learner.collected_rewards)    
    print(e)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS","UCB"])
plt.show()

x=np.arange(50,700,0.01)
for _ in range(len(features_column_to_class)):  
    for i in range(n_arms):
        plt.plot(x, norm.pdf(x, context_gts_learner.learners[_].means_of_rewards[i], 1/context_gts_learner.learners[_].precision_of_rewards[i]), label=str(i))
    plt.legend()
    plt.show()


