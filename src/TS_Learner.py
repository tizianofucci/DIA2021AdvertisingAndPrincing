from Learner import *

# Thompson sampling
# associates with each arm a beta distribution.
# the distribution for each arm will be updated based on the received reward when pulled
class TS_Learner(Learner):
    
    # each arm is initially associated with a (1,1) beta distribution.
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    # pulled arm is the one with the highest alpha param among the beta distributions.
    def pull_arm(self):
        pulled_arm = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        return pulled_arm
    
    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm,0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm,1] + 1.0 - reward