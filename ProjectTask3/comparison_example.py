from queue import Queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import seaborn as sns
from Environment import *
from PricingEnvironment import *
from TS_Learner import *
from GaussianTS_Learner import *
from Greedy_Learner import *
from UCB_Learner import *
import math
import scipy.stats as stats
from scipy.stats import norm

n_arms = 10
p = np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
prices = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])


T = 5000
n_experiment = 20
delay = 30
mu_new_customer = 12
sigma_new_customer = math.sqrt(4)

opt = p[1]*prices[1]*mu_new_customer


#mu = 0
#variance = 1
#sigma = math.sqrt(variance)
#x = np.linspace(mu_new_customer - 3*sigma_new_customer, mu_new_customer + 3*sigma_new_customer, 100)
#plt.plot(x, stats.norm.pdf(x, mu_new_customer, sigma_new_customer))
#plt.show()




ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []
pulled_arm_buffer_ts = Queue(maxsize=31)
pulled_arm_buffer_ucb = Queue(maxsize=31)


for e in range(0,n_experiment):
    env = PricingEnvironment(n_arms,prices,p,mu_new_customer,sigma_new_customer)
#    ts_learner = TS_Learner(n_arms)
    gts_learner = GaussianTS_Learner(n_arms,delay)
    ucb_learner = UCB_Learner(n_arms,delay)
    pulled_arm_buffer_ts.queue.clear()
    pulled_arm_buffer_ucb.queue.clear()

    for t in range (0,T):
#        pulled_arm_buffer_ts.put(ts_learner.pull_arm())
        pulled_arm_buffer_ts.put(gts_learner.pull_arm())
        pulled_arm_buffer_ucb.put(ucb_learner.pull_arm())

        if t>=delay:
#            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
#            rewards = env.round(after_30_days_arm_ts)
#            ts_learner.update(after_30_days_arm_ts,rewards)

            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts)
            gts_learner.update(after_30_days_arm_ts,rewards)

            after_30_days_arms_ucb = pulled_arm_buffer_ucb.get()
            rewards = env.round(after_30_days_arms_ucb)
            ucb_learner.update(after_30_days_arms_ucb,rewards)
    ts_rewards_per_experiment.append(gts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
    print(e)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS","UCB"])
plt.show()

ax = plt.subplots()
ax = sns.barplot(x=np.array(list("ABCDEFGHIJ")),y=ucb_learner.upper_bounds,color='r')
ax = sns.barplot(x=np.array(list("ABCDEFGHIJ")),y=ucb_learner.means,color='b')

plt.show()



x=np.arange(1,80,0.01)
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[0], 1/gts_learner.precision_of_rewards[0]), label='0')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[1], 1/gts_learner.precision_of_rewards[1]), label='1')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[2], 1/gts_learner.precision_of_rewards[2]), label='2')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[3], 1/gts_learner.precision_of_rewards[3]), label='3')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[4], 1/gts_learner.precision_of_rewards[4]), label='4')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[5], 1/gts_learner.precision_of_rewards[5]), label='5')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[6], 1/gts_learner.precision_of_rewards[6]), label='6')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[7], 1/gts_learner.precision_of_rewards[7]), label='7')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[8], 1/gts_learner.precision_of_rewards[8]), label='8')
plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[9], 1/gts_learner.precision_of_rewards[9]), label='9')


plt.legend()
plt.show()
