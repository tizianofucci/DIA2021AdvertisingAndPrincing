from PricingEnvironment import PricingEnvironment
import numpy as np

class ContextEnvironment():
    def __init__(self,n_arms,prices, prod_cost, contexts_prob,context_bid_offsets, contexts_mu,contexts_sigma,contexts_n_returns_coeff,features_matrix):
        self.environments = [PricingEnvironment(n_arms,prices, prod_cost, contexts_prob, context_bid_offsets, contexts_mu,contexts_sigma,contexts_n_returns_coeff,features_matrix) for _ in range(4)]

    def round(self,after_30_days_arm_ts,bid):
        rewards = [np.array([]) for i in range(4)]
        users_segmentation = [np.array([]) for i in range(4)]
        rewards[0],users_segmentation[0] = self.environments[0].round([after_30_days_arm_ts[0], after_30_days_arm_ts[0], after_30_days_arm_ts[0], after_30_days_arm_ts[0]],bid)
        rewards[1],users_segmentation[1] = self.environments[1].round([after_30_days_arm_ts[1], after_30_days_arm_ts[1], after_30_days_arm_ts[2], after_30_days_arm_ts[2]],bid)
        rewards[2],users_segmentation[2] = self.environments[2].round([after_30_days_arm_ts[3], after_30_days_arm_ts[4], after_30_days_arm_ts[3], after_30_days_arm_ts[4]],bid)
        rewards[3],users_segmentation[3] = self.environments[3].round([after_30_days_arm_ts[5], after_30_days_arm_ts[6], after_30_days_arm_ts[7], after_30_days_arm_ts[8]],bid)


        return rewards,users_segmentation