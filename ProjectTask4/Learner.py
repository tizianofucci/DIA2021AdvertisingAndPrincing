import numpy as np

class Learner():
    def __init__(self, n_arms) :
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [np.array([]) for i in range (n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self,pulled_arm,rewards):
        self.rewards_per_arm[pulled_arm] = np.append(self.rewards_per_arm[pulled_arm],rewards)
        self.collected_rewards = np.append(self.collected_rewards,rewards)
        
        
