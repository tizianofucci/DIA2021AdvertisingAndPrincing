from Learner import *
import numpy as np

class GaussianTS_Learner(Learner):
    def __init__(self, n_arms,delay):
        super().__init__(n_arms)

        self.τ_0 = 0.0001  # the posterior precision
        self.μ_0 = 1       # the posterior mean
        self.τ = 0.1
        self.means_of_rewards = [1 for _ in range(self.n_arms)]
        self.precision_of_rewards = [0.0001 for _ in range(self.n_arms)]
        self.delay = delay

    def pull_arm(self):
        if(self.t < self.n_arms + self.delay):
            idx = self.t % self.n_arms
        else :
            idx = np.argmax(np.random.normal(self.means_of_rewards,np.divide(1, np.sqrt(self.precision_of_rewards))))
        self.t += 1
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.means_of_rewards[pulled_arm] = ((self.precision_of_rewards[pulled_arm] * self.means_of_rewards[pulled_arm]) + (self.τ * len(self.rewards_per_arm[pulled_arm]) * np.mean(self.rewards_per_arm[pulled_arm])))/(self.precision_of_rewards[pulled_arm] + self.τ * len(self.rewards_per_arm[pulled_arm]))
        #self.means_of_rewards[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        #self.std_reward[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm])
        self.precision_of_rewards[pulled_arm] += self.τ