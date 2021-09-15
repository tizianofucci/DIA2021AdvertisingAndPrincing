import numpy as np

class MatrixLearner():
    def __init__(self, n_arms0, n_arms1) :
        self.n_arms0 = n_arms0
        self.n_arms1 = n_arms1
        self.idx_price = 1
        self.idx_bid = 0
        self.t = 0
        self.rewards_per_arm = x = [[np.array([]) for j in range(n_arms1)] for i in range (n_arms0)]
        self.collected_rewards = np.array([])

    def update_observations(self,pulled_arm,rewards):
        self.rewards_per_arm[pulled_arm[self.idx_bid]][pulled_arm[self.idx_price]] = np.append(self.rewards_per_arm[pulled_arm[self.idx_bid]][pulled_arm[self.idx_price]],rewards)
        self.collected_rewards = np.append(self.collected_rewards,rewards)
        
        
