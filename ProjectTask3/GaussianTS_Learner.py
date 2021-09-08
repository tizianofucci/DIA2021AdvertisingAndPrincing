from Learner import *
import numpy as np

class GaussianTS_Learner(Learner):
    def __init__(self, n_arms,delay):
        super().__init__(n_arms)
        self.means_of_rewards = [0 for _ in range(self.n_arms)]
        self.std_reward = [float("+inf") for _ in range(self.n_arms)]
        self.delay = delay

    def pull_arm(self):
        if(self.t < self.n_arms + self.delay):
            idx = self.t % self.n_arms
        else :
            idx = np.argmax(np.random.normal(self.means_of_rewards,self.std_reward))
        self.t += 1
        return idx

    def update(self, pulled_arm, reward):        
        self.update_observations(pulled_arm, reward)
        self.means_of_rewards[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        self.std_reward[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm])