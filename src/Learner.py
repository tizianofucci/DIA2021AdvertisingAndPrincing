import numpy as np



# Learner class, abstract class
#
# n_arms: n of arms on which to experiment
# t : time step
# rewards_per_arm: list of lists of rewards for each arm. 
#   each pull will append the reward to the list for the corresponding arm
# collected_rewards: list of all collected rewards, regardless of arm pull.
class Learner:    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0 
        self.rewards_per_arm = [[] for i in range(n_arms)] #list of lists, initialized at list of empty lists
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)