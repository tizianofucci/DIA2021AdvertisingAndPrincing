import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import math
import matplotlib.pyplot as plt
from queue import Queue
from scipy.stats import norm

from BiddingEnvironment import *
from GaussianTS_Learner import *

n_arms = 10

price = 6.5 #fixed
prod_cost = 3.0
conv_rate = 0.4 #prob of buying after click, known

T = 365
n_experiment = 100
delay = 30
mu_new_customer = 12
sigma_new_customer = math.sqrt(1)


#bids = np.array([0.5,0.7,0.9,1.1,1.4,1.5,1.7,1.9,2.1,2.3])
#bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])
bids = np.array([0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7])

#bid_modifiers = np.array([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,1.8])
bid_modifiers = np.array([0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4])

def expected(arm):
    bid = bids[arm]
    delta_customers = 200*(bid_modifiers[arm]*2)
    return conv_rate*(mu_new_customer + delta_customers)*(price - prod_cost)*(3.0/(2*(price - 3.5)) + 1) - (bid - bid/10) * (mu_new_customer + delta_customers)

expected_rewards = [expected(x) for x in range(n_arms)]
print("expected rewards:\n", expected_rewards)

opt_arm = argmax(expected_rewards)
opt = expected_rewards[opt_arm]


ts_rewards_per_experiment = []

gts_means_of_rewards = []
gts_precision_of_rewards = []


pulled_arm_buffer_ts = Queue(maxsize=31)
pulled_arm_buffer_ucb = Queue(maxsize=31)

opt_count = 0

for e in range(0,n_experiment):
    env = BiddingEnvironment(n_arms, price, prod_cost, bids, bid_modifiers, conv_rate, mu_new_customer, sigma_new_customer)
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

    #print(env.results)
    print(e)
    n_pulls_per_arm = [len(x) for x in gts_learner.rewards_per_arm]
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
gts_means_of_rewards = np.transpose(gts_means_of_rewards)
gts_precision_of_rewards = np.transpose(gts_precision_of_rewards)


x=np.arange(-200,200,0.01)
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

#for i in range(len(gts_learner.means_of_rewards)):
    #print("mean: {}, cdf in 0: {}".format(gts_learner.means_of_rewards[i],norm.cdf(0, gts_learner.means_of_rewards[i], 1/gts_learner.precision_of_rewards[i])))

plt.legend()
plt.show()



#x=np.arange(-200,200,0.01)
#for i in range(n_arms):
#    plt.plot(x, norm.pdf(x, np.mean(gts_means_of_rewards[i]), np.mean(1/gts_precision_of_rewards[i])), label="{}".format(i))

#plt.legend()
#plt.show()

x=np.arange(-200,200,0.01)
for i in range(n_arms):
    variance = (np.mean((1/gts_precision_of_rewards[i])**2)) + np.mean(gts_means_of_rewards[i]**2 - (np.mean(gts_means_of_rewards[i]))**2)
    plt.plot(x, norm.pdf(x, np.mean(gts_means_of_rewards[i]), math.sqrt(variance)), label="{}".format(i))


#for i in range(len(gts_learner.means_of_rewards)):
    #print("mean: {}, cdf in 0: {}".format(gts_learner.means_of_rewards[i],norm.cdf(0, gts_learner.means_of_rewards[i], 1/gts_learner.precision_of_rewards[i])))

plt.legend()
plt.show()