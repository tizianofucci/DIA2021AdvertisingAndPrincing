from Learner import *
import numpy as np
from GaussianTS_Learner import GaussianTS_Learner

class ContextGaussianTS_Learner():
    def __init__(self, n_arms,delay,features_matrix):
        self.active_learners = range(5,9)
        self.collected_rewards = np.array([])
        self.features_matrix = features_matrix
        #TODO parametrize the 9
        self.learners = [GaussianTS_Learner(n_arms,delay) for _ in range(9)]

        self.mu_class_prob = np.zeros(len(features_matrix))
        self.class_count = np.zeros(len(features_matrix))

    def pull_arm(self):
        indexes = np.zeros(len(self.learners),dtype=int)
        for i in range(len(indexes)):
            indexes[i] = self.learners[i].pull_arm()
        return indexes

    def update(self, pulled_arm, reward,users_segmentation):   
        actual_rewards = np.zeros(9)

        actual_rewards[0] = np.sum(reward[0])
        actual_rewards[1] = reward[1][0] + reward[1][1]   
        actual_rewards[2] = reward[1][2] + reward[1][3]     
        actual_rewards[3] = reward[2][0] + reward[2][2]     
        actual_rewards[4] = reward[2][1] + reward[2][3]     
        actual_rewards[5] = reward[3][0]
        actual_rewards[6] = reward[3][1]     
        actual_rewards[7] = reward[3][2]      
        actual_rewards[8] = reward[3][3] 

        for i in range(9):
            self.learners[i].update(pulled_arm[i],actual_rewards[i])   
    
     
        self.class_count = self.class_count + users_segmentation
        self.mu_class_prob = self.class_count / sum(self.class_count)
        self.collected_rewards = np.append(self.collected_rewards,np.sum(actual_rewards[self.active_learners]))



    def try_splitting(self,users_segmentation):
        pass