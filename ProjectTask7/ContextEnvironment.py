from PricingBiddingEnvironment import PricingBiddingEnvironment
import numpy as np

class ContextEnvironment():
    def __init__(self,prices,prod_cost,bids,bid_modifiers,contexts_bid_offsets, contexts_prob,contexts_mu,contexts_sigma, delta_customers_multipliers, contexts_n_returns_coeffs, features_matrix):
        self.environments = [PricingBiddingEnvironment(prices,prod_cost,bids,bid_modifiers,contexts_bid_offsets, contexts_prob,contexts_mu,contexts_sigma, delta_customers_multipliers, contexts_n_returns_coeffs, features_matrix) for _ in range(4)]

    def round(self,after_30_days_arm_ts):
        rewards = [np.array([]) for i in range(4)]
        users_segmentation = [np.array([]) for i in range(4)]
        rewards[0],users_segmentation[0] = self.environments[0].round([[after_30_days_arm_ts[0][0], after_30_days_arm_ts[0][1]], [after_30_days_arm_ts[0][0], after_30_days_arm_ts[0][1]], [after_30_days_arm_ts[0][0], after_30_days_arm_ts[0][1]], [after_30_days_arm_ts[0][0], after_30_days_arm_ts[0][1]]])
        rewards[1],users_segmentation[1] = self.environments[1].round([[after_30_days_arm_ts[1][0], after_30_days_arm_ts[1][1]], [after_30_days_arm_ts[1][0], after_30_days_arm_ts[1][1]], [after_30_days_arm_ts[2][0], after_30_days_arm_ts[2][1]], [after_30_days_arm_ts[2][0], after_30_days_arm_ts[2][1]]])
        rewards[2],users_segmentation[2] = self.environments[2].round([[after_30_days_arm_ts[3][0], after_30_days_arm_ts[3][1]], [after_30_days_arm_ts[4][0], after_30_days_arm_ts[4][1]], [after_30_days_arm_ts[3][0], after_30_days_arm_ts[3][1]], [after_30_days_arm_ts[4][0], after_30_days_arm_ts[4][1]]])
        rewards[3],users_segmentation[3] = self.environments[3].round([[after_30_days_arm_ts[5][0], after_30_days_arm_ts[5][1]], [after_30_days_arm_ts[6][0], after_30_days_arm_ts[6][1]], [after_30_days_arm_ts[7][0], after_30_days_arm_ts[7][1]], [after_30_days_arm_ts[8][0], after_30_days_arm_ts[8][1]]])


        return rewards,users_segmentation