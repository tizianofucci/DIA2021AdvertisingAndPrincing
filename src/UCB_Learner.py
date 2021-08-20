from numpy import log
from Learner import *
import math


class UCB_Learner(Learner):
    
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.upper_bounds = np.zeros(n_arms)
        self.means = np.zeros(n_arms)

    # pulled arm is the one with the highest upper confidence bound.
    def pull_arm(self):
        for i in range(self.n_arms):
            self.upper_bounds[i] = self.means[i] + math.sqrt(  (2*log(self.t)) / (len(self.rewards_per_arm[i])))
        
        pulled_arm = np.argmax(self.upper_bounds)
        return pulled_arm
    
    def update(self, pulled_arm, reward):
        self.t+=1
        self.means[pulled_arm] = ( (self.means[pulled_arm]*(len(self.rewards_per_arm[pulled_arm]))) + reward ) / (len(self.rewards_per_arm[pulled_arm]) + 1)
        self.update_observations(pulled_arm, reward)
