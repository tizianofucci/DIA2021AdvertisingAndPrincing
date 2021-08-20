import numpy as np
import matplotlib.pyplot as plt

from Environment import *
from TS_Learner import *
from UCB_Learner import *

n_arms = 10
p = np.array([0.15, 0.1, 0.1, 0.35, 0.4, 0.2, 0.3, 0.7, 0.6, 0.5])

opt = p[7]

T = 365 ##time horizon

n_experiments = 1000

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []


for e in range (0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms)
    ucb_learner = UCB_Learner(n_arms=n_arms)

    for i in range (0, T):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)


    #initialize ucb learner
    for i in range(n_arms):
        reward = env.round(i)
        ucb_learner.update(i, reward)

    for i in range(n_arms, T):
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
    
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'g')

plt.legend(["TS","UCB1"])

#to compare with rough approx of theoretical upper bounds..
#x = np.linspace(1, 300)
#plt.plot(x, log(x) + log(log(x)))
#plt.plot(x, 4*log(x))
#plt.legend(["TS","UCB1", "bound_ts", "bound_ucb"])

N = 10
means = p
bounds = ucb_learner.upper_bounds
emp_means = ucb_learner.means
ind = np.arange(N)
fig = plt.figure(1)
ax = fig.add_subplot()

ax.bar(ind, bounds, width=0.05, color='black')
ax.bar(ind, means, width=0.25, color='g')
ax.bar(ind, emp_means, width=0.15, color='r')
ax.set_yticks(np.arange(0, 1, 0.05))
ax.set_xticks(ind)
ax.legend(labels=['UCB1_bound', 'mean',  'emp_means'])
for i in range(n_arms):
    print(len(ucb_learner.rewards_per_arm[i]))
plt.tight_layout()
plt.show()
