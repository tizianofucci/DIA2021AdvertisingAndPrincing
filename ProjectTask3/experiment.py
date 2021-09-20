import os, sys 
currentdir = os.path.dirname(os.path.realpath(__file__)) 
parentdir = os.path.dirname(currentdir) 
sys.path.append(parentdir)
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Environment import *
from PricingEnvironment import *
from GaussianTS_Learner import *
from UCB_Learner import *
import math
from math import e
from scipy.stats import norm
from UtilFunctions import *
import UtilFunctions


n_arms = 10
prod_cost = 3.0
T = 365
n_experiment = 50
delay = 30      #timesteps of delay before reward is discovered
sigma_new_customer = math.sqrt(1)

bids = UtilFunctions.global_bids
bid_modifiers_c1 = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.4, 1.4]
bid_modifiers_c2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bid_modifiers_c3 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

bid_modifiers = [bid_modifiers_c1, bid_modifiers_c2, bid_modifiers_c3]
bid_idx = 8


bid = bids[bid_idx]

class_deltas = [] # New clicks for each class. depends on selected bid (fixed in this task).
for j in bid_modifiers:
    class_deltas.append(compute_delta_customers(j[bid_idx]))    

class_mu_base = [10, 10, 10]
class_mu = np.add(class_mu_base, class_deltas)
mu_new_customer = np.sum(class_mu) # Number of total clicks = sum of new clicks from each class 

n_arms = 10

prices = UtilFunctions.global_prices
class_probs = [[conv_c1(x) for x in prices],[conv_c2(x) for x in prices],[conv_c3(x) for x in prices]]
conv_rates = np.average(class_probs, axis=0, weights = class_mu)

coeffs = [4.0, 2.0, 3.0]
avg_coeff = np.average(coeffs, weights = class_mu)  

bid_offsets = [8.0, 5.0, 15.0]
avg_bid_offset = np.average(bid_offsets, weights = class_mu)  


"""
Computes expected reward given an arm.

"""
def expected(arm):
    price = prices[arm]
    expected_returns = (avg_coeff/(2*((price)/10)+0.5))  
    return (conv_rates[arm]*(price - prod_cost) * mu_new_customer * (expected_returns + 1)) - (bid - bid/avg_bid_offset) * mu_new_customer

expected_rewards = [expected(x) for x in range(n_arms)]

opt_arm = np.argmax(expected_rewards)
opt = expected_rewards[opt_arm]
print("Optimal reward: {}; with arm {}".format(opt, opt_arm))

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

pulled_arm_buffer_ts = Queue(maxsize=31)
pulled_arm_buffer_ucb = Queue(maxsize=31)


"""
Run experiments, collect rewards
"""
for e in range(0,n_experiment):
    env = PricingEnvironment(n_arms,prices,prod_cost,conv_rates,mu_new_customer,sigma_new_customer, returns_coeff=avg_coeff, bid_offset=avg_bid_offset)
    gts_learner = GaussianTS_Learner(n_arms,delay)
    ucb_learner = UCB_Learner(n_arms,delay)
    pulled_arm_buffer_ts.queue.clear()
    pulled_arm_buffer_ucb.queue.clear()

    for t in range (0,T):
        pulled_arm_buffer_ts.put(gts_learner.pull_arm())
        pulled_arm_buffer_ucb.put(ucb_learner.pull_arm())

        if t>=delay:
            after_30_days_arm_ts = pulled_arm_buffer_ts.get()
            rewards = env.round(after_30_days_arm_ts,bid)
            gts_learner.update(after_30_days_arm_ts,rewards)

            after_30_days_arms_ucb = pulled_arm_buffer_ucb.get()
            rewards = env.round(after_30_days_arms_ucb,bid)
            ucb_learner.update(after_30_days_arms_ucb,rewards)
    ts_rewards_per_experiment.append(gts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
    
    print(e)



## Plot cumulative regret results

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r', linewidth=4)
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'g', linewidth=4)
plt.legend(["TS","UCB"])
plt.show()



## Plot daily rewards

x=np.arange(0,335,1)
plt.xlabel("t")
plt.ylabel("Rewards - TS")
plt.plot(x, np.mean(ts_rewards_per_experiment, axis=0),'-ok',color='red', markersize=4, linewidth=0.25)
plt.show()

plt.xlabel("t")
plt.ylabel("Rewards - UCB")
plt.plot(x, np.mean(ucb_rewards_per_experiment, axis=0),'-ok',color='green',markersize=4, linewidth=0.15)
plt.show()



## Plot UCB arms' means and upper bound pair.

ax = plt.subplots()
ax = sns.barplot(x=np.array(UtilFunctions.global_prices),y=ucb_learner.upper_bounds,color='r')
ax = sns.barplot(x=np.array(UtilFunctions.global_prices),y=ucb_learner.means,color='b')
plt.show()



#Plot TS learned gaussians

x=np.arange(100,(opt*1.2),0.01)
for i in range(n_arms):
    plt.plot(x, norm.pdf(x, gts_learner.means_of_rewards[i], 1/np.sqrt(gts_learner.precision_of_rewards[i])), label=str(i))
plt.legend()
plt.xlabel("X")
plt.ylabel("P(X)")
plt.show()
