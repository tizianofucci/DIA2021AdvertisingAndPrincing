from numpy import log, sqrt
import numpy as np
from Learner import *

class UCB_Learner(Learner):
    def __init__(self, n_arms,delay):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.upper_bounds = np.zeros(n_arms)
        self.delay = delay

    ### Loops over all arms until first reward is discovered for every arm (n_arms + delay)
    def pull_arm(self):
        if(self.t < self.n_arms + self.delay):
            return_index = self.t % self.n_arms
        else :
            return_index = np.argmax(self.upper_bounds)
        self.t += 1
        return return_index

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        for i in range(0,self.n_arms):
            if len(self.rewards_per_arm[i]) > 0:
                ucb_coeff = 80  # Multiplier used for rescaling the upper bound. Used for tuning exploration/exploitation.
                self.upper_bounds[i] = self.means[i] + ucb_coeff*sqrt((2*log(self.t+1))/len(self.rewards_per_arm[i]))
            else:
                self.upper_bounds[i] = 0
