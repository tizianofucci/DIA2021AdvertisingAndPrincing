import numpy as np


#The environment in which experiments are ran
#n_arms: number of arms
#probabilities: array of probability distributions, one for each arm.
class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
    
    #we can draw from the distribution with the pulled_arm index at each round.
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm]) #(desired draws, success probability)
        return reward