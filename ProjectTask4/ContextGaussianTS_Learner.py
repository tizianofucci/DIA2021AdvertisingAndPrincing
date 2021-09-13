from Learner import *
import numpy as np
from GaussianTS_Learner import GaussianTS_Learner

class ContextGaussianTS_Learner():
    def __init__(self, n_arms,delay,features_matrix):
        self.active_learners = [True,False,False,False,False,False,False,False,False]
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
    
     
        self.class_count = self.class_count + sum(users_segmentation)
        self.mu_class_prob = self.class_count / sum(self.class_count)
        self.collected_rewards = np.append(self.collected_rewards,np.sum(actual_rewards[self.active_learners]))



    def try_splitting(self):
        if self.active_learners == [True,False,False,False,False,False,False,False,False]:
            split_a = np.mean(self.learners[1].collected_rewards)-np.std(self.learners[1].collected_rewards) + np.mean(self.learners[2].collected_rewards) - np.std(self.learners[2].collected_rewards)
            split_b = np.mean(self.learners[3].collected_rewards)-np.std(self.learners[3].collected_rewards) + np.mean(self.learners[4].collected_rewards) - np.std(self.learners[4].collected_rewards)

            proposed_split = np.max([split_a, split_b])

            if proposed_split > (np.mean(self.learners[0].collected_rewards) - np.std(self.learners[0].collected_rewards)):
                if proposed_split == split_a:
                    print("split su a")
                    self.active_learners = [False,True,True,False,False,False,False,False,False]
#                    self.active_learners = [False,False,False,False,False,True,True,True,True]
                if proposed_split == split_b:
                    print("split su b")
                    self.active_learners = [False,False,False,True,True,False,False,False,False]
#                    self.active_learners = [False,False,False,False,False,True,True,True,True]

        elif self.active_learners == [False,True,True,False,False,False,False,False,False]:
            split_a = np.mean(self.learners[5].collected_rewards)-np.std(self.learners[5].collected_rewards) + np.mean(self.learners[6].collected_rewards)-np.std(self.learners[6].collected_rewards)
            split_b = np.mean(self.learners[7].collected_rewards)-np.std(self.learners[7].collected_rewards) + np.mean(self.learners[8].collected_rewards)-np.std(self.learners[8].collected_rewards)

            if split_a > np.mean(self.learners[1].collected_rewards) - np.std(self.learners[1].collected_rewards) :
                self.active_learners = [False,False,True,False,False,True,True,False,False]
                print("split su b in 1, dopo a")

            if split_b > np.mean(self.learners[2].collected_rewards) - np.std(self.learners[2].collected_rewards) :
                print("split su b in 1, dopo a")
                self.active_learners = [False,True,False,False,False,False,False,True,True]

        elif self.active_learners == [False,False,False,True,True,False,False,False,False]:
            split_a = np.mean(self.learners[6].collected_rewards)-np.std(self.learners[6].collected_rewards) + np.mean(self.learners[8].collected_rewards)-np.std(self.learners[8].collected_rewards)
            split_b = np.mean(self.learners[5].collected_rewards)-np.std(self.learners[5].collected_rewards)+ np.mean(self.learners[7].collected_rewards)-np.std(self.learners[7].collected_rewards)

            if split_a > np.mean(self.learners[3].collected_rewards)-np.std(self.learners[3].collected_rewards) :
                print("split su a in 0, dopo b")
                self.active_learners = [False,False,False,True,False,False,True,False,True]
            if split_b > np.mean(self.learners[4].collected_rewards)-np.std(self.learners[4].collected_rewards) :
                print("split su a in 1, dopo b")
                self.active_learners = [False,False,False,False,True,True,False,True,False]
        elif self.active_learners != [False,False,False,False,False,True,True,True,True]:

            complete_split = np.mean(self.learners[5].collected_rewards)-np.std(self.learners[5].collected_rewards) +np.mean(self.learners[6].collected_rewards)-np.std(self.learners[6].collected_rewards) + np.mean(self.learners[7].collected_rewards)-np.std(self.learners[7].collected_rewards) + np.mean(self.learners[8].collected_rewards)-np.std(self.learners[8].collected_rewards)
            if complete_split > np.mean(self.collected_rewards)-np.std(self.collected_rewards):
                print("Final Split")
                self.active_learners = [False,False,False,False,False,True,True,True,True]
