from Learner import *
import numpy as np
from GaussianTS_Learner import GaussianTS_Learner


"""
Class containing 9 learners that train simultaneously in order to figure out
the best contexts to learn
"""
class ContextGaussianTS_Learner():
    def __init__(self, n_arms,delay,features_matrix):
        self.n_learners = 9
        self.active_learners = [True,False,False,False,False,False,False,False,False]
        self.collected_rewards = np.array([])
        self.features_matrix = features_matrix
        #TODO parametrize the 9
        self.learners = [GaussianTS_Learner(n_arms,delay) for _ in range(self.n_learners)]

    def pull_arm(self):
        indexes = np.zeros(len(self.learners),dtype=int)
        for i in range(len(indexes)):
            indexes[i] = self.learners[i].pull_arm()
        return indexes

    """
    Given the rewards from the independent environments,
    they are used to update the relative learner
    """
    def update(self, pulled_arm, reward):   
        actual_rewards = np.zeros(self.n_learners)

        actual_rewards[0] = np.sum(reward[0])
        actual_rewards[1] = reward[1][0] + reward[1][1]   
        actual_rewards[2] = reward[1][2] + reward[1][3]     
        actual_rewards[3] = reward[2][0] + reward[2][2]     
        actual_rewards[4] = reward[2][1] + reward[2][3]     
        actual_rewards[5] = reward[3][0]
        actual_rewards[6] = reward[3][1]     
        actual_rewards[7] = reward[3][2]      
        actual_rewards[8] = reward[3][3] 

        for i in range(self.n_learners):
            self.learners[i].update(pulled_arm[i],actual_rewards[i])   
        self.collected_rewards = np.append(self.collected_rewards,np.sum(actual_rewards[self.active_learners]))

    def lower_bound(self,id,best_arms):        
        return self.learners[id].means_of_rewards[best_arms[id]] - 18 * np.sqrt(1/self.learners[id].precision_of_rewards[best_arms[id]])

    def is_active(self, idx):
        return (self.active_learners[idx] == True)


    """
    Verifies if the conditions for context splitting are met, it uses a lower confidence bound on the best arm
    """
    def try_splitting(self):
        best_arms = [np.argmax(self.learners[i].means_of_rewards) for i in range(self.n_learners)]

        if self.active_learners == [True,False,False,False,False,False,False,False,False]:
            split_a = self.lower_bound(1,best_arms) + self.lower_bound(2,best_arms)
            split_b = self.lower_bound(3,best_arms) + self.lower_bound(4,best_arms)

            proposed_split = np.max([split_a, split_b])

            if proposed_split > self.lower_bound(0,best_arms):
                if proposed_split == split_a:
                    print("Learner attivati : 0x 1x (OK)")
                    self.active_learners = [False,True,True,False,False,False,False,False,False]
                if proposed_split == split_b:
                    print("Learner attivati : x0 x1")
                    self.active_learners = [False,False,False,True,True,False,False,False,False]

        elif self.active_learners == [False,True,True,False,False,False,False,False,False]:
            split_a = self.lower_bound(5,best_arms) + self.lower_bound(6,best_arms)
            split_b = self.lower_bound(7,best_arms) + self.lower_bound(8,best_arms)
            
            diff_a = (split_a - self.lower_bound(1,best_arms))
            diff_b = (split_b - self.lower_bound(2,best_arms))
            
            
            if diff_a>diff_b and diff_a > 0 :
                self.active_learners = [False,False,True,False,False,True,True,False,False]
                print("Learner attivati : 00 01 1x")

            elif diff_b > 0 :
                print("Learner attivati : 0x 10 11 (OKOK)")
                self.active_learners = [False,True,False,False,False,False,False,True,True]

        elif self.active_learners == [False,False,False,True,True,False,False,False,False]:
            split_a = self.lower_bound(6,best_arms) + self.lower_bound(8,best_arms)
            split_b = self.lower_bound(5,best_arms) + self.lower_bound(7,best_arms)

            diff_a = (split_a - self.lower_bound(4,best_arms))
            diff_b = (split_b - self.lower_bound(3,best_arms))

            if diff_a > diff_b and diff_a > 0:
                print("Learner attivati : x0 01 11")
                self.active_learners = [False,False,False,True,False,False,True,False,True]
            elif diff_b > 0 :
                print("Learner attivati : 00 10 x1")
                self.active_learners = [False,False,False,False,True,True,False,True,False]
        elif self.active_learners != [False,False,False,False,False,True,True,True,True]:

            complete_split = self.lower_bound(5,best_arms) + self.lower_bound(6,best_arms)+ self.lower_bound(7,best_arms) + self.lower_bound(8,best_arms)
            active_learners_bound = 0
            for i in range(self.n_learners):
                if self.active_learners[i] == True:
                    active_learners_bound += self.lower_bound(i,best_arms)
            if complete_split > active_learners_bound:
                print("Final Split")
                self.active_learners = [False,False,False,False,False,True,True,True,True]
