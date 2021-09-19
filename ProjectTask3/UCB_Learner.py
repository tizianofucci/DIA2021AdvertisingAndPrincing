from numpy import log, mod, sqrt
import numpy
from pandas.core.dtypes.missing import na_value_for_dtype
from Learner import *

class UCB_Learner(Learner):
    def __init__(self, n_arms,delay):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.upper_bounds = np.zeros(n_arms)
        self.delay = delay

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
                self.upper_bounds[i] = self.means[i] + 150*sqrt((2*log(self.t+1))/len(self.rewards_per_arm[i]))
            else:
                self.upper_bounds[i] = 0
