from scipy.sparse.construct import rand
from Learner import *
import numpy as np
import scipy.stats
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, CompoundKernel as C

class GPTS_Learner(Learner):
    def __init__(self, n_arms_bids, n_arms_prices, arms, delay):
        super().__init__(n_arms_bids*n_arms_prices)
        self.idx_price = 1
        self.idx_bid = 0
        self.n_arms_bids = n_arms_bids
        self.n_arms_prices = n_arms_prices
        self.pulled_arms = []
        self.arms = arms
        self.means_of_rewards = np.zeros((self.n_arms_bids, self.n_arms_prices))
        self.sigmas = np.ones((self.n_arms_bids, self.n_arms_prices))
        self.delay = delay

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, normalize_y=False, n_restarts_optimizer= 9)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms[0].append(self.arms[pulled_arm[0]])
        self.pulled_arms[1].append(self.arms[pulled_arm[1]])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms), returdn_std = True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def pull_arm(self):
        # estimate probability of negative revenue for each arm, if > 20% don't consider eligible for pulling.
        mask = np.zeros((self.n_arms_bids, self.n_arms_prices))
        
        if(self.t < self.n_arms + self.delay + 10):
            idx0 = random.randrange(self.n_arms_bids)
            idx1 = random.randrange(self.n_arms_prices)
            idx = (idx0, idx1)
        else :
            for ix,iy in np.ndindex(mask.shape):
                neg_revenue_estimate = scipy.stats.norm.cdf(0, self.means_of_rewards[ix, iy], np.divide(1, self.precision_of_rewards[ix, iy]))
                if(neg_revenue_estimate > 0.2):
                    mask[ix, iy] = 1
                if (sum(mask) == mask.size):
                    raise Exception('negative revenue on all arms')
            samples = np.random.normal(self.means_of_rewards, self.sigmas)
            masked_samples = np.ma.masked_array(samples, mask)
            idx = np.unravel_index(np.argmax(masked_samples, axis=None), masked_samples.shape)
        self.t += 1
        return idx

    def update(self, pulled_arm, reward):        
        self.update_observations(pulled_arm, reward)
        self.update_model()