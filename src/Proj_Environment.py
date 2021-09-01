import numpy as np

class Proj_Environment:
    def __init__(self, n_arms, arms):
        self.n_arms = n_arms
        self.arms = arms
    
    def round(self, pulled_arm):
        n_returns = np.random.poisson(1, (3.0/(2*(pulled_arm-3.5))))
        #inst_reward = np.random * pulled_arm 
        total_reward = pulled_arm * (1 + n_returns)
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward