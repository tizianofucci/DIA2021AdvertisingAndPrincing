
from scipy.sparse.construct import rand
from MatrixLearner import *
import numpy as np
import scipy.stats
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(MatrixLearner):
    def __init__(self, n_arms_bids, n_arms_prices, arms, delay):
        super().__init__(n_arms_bids,n_arms_prices)
        self.idx_price = 1
        self.idx_bid = 0
        self.n_arms_bids = n_arms_bids
        self.n_arms_prices = n_arms_prices
        self.pulled_arms = np.array([])
        self.arms = arms
        self.matrixArms = [(i, j) for i in self.arms[0] for j in self.arms[1]]
        self.means = np.zeros((self.n_arms_bids, self.n_arms_prices))
        self.sigmas = np.ones((self.n_arms_bids, self.n_arms_prices))
        self.delay = delay

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, normalize_y=False, n_restarts_optimizer= 9)

    def update_observations_gp(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms,[self.arms[self.idx_bid][pulled_arm[self.idx_bid]], self.arms[self.idx_price][pulled_arm[self.idx_price]]])

    def update_model(self):

        x = np.array(self.pulled_arms).T
        x = np.reshape(self.pulled_arms, (-1,2))
        y = self.collected_rewards
        self.gp.fit(x,y)


        self.means, self.sigmas = self.gp.predict(self.matrixArms, return_std=True)
#        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def pull_arm(self):
        # estimate probability of negative revenue for each arm, if > 20% don't consider eligible for pulling.
        mask = np.zeros(self.n_arms_bids* self.n_arms_prices)
        
        if(self.t < self.delay + 10):
            idx0 = random.randrange(self.n_arms_bids)
            idx1 = random.randrange(self.n_arms_prices)
            idx = (idx0, idx1)
        else :
            for i in range(len(mask)):
                neg_revenue_estimate = scipy.stats.norm.cdf(0, self.means[i], self.sigmas[i])
                if(neg_revenue_estimate > 0.2):
                    mask[i] = 1
                if (sum(mask) == mask.size):
                    raise Exception('negative revenue on all arms')
            masked_means = np.ma.masked_array(self.means, mask)
            masked_sigmas = np.ma.masked_array(self.sigmas, mask)
            samples = np.random.normal(masked_means, masked_sigmas)
            idx = np.unravel_index(np.argmax(samples, axis=None), (self.n_arms_bids,self.n_arms_prices))
        self.t += 1
        return idx

    def update(self, pulled_arm, reward):        
        self.update_observations_gp(pulled_arm, reward)
        self.update_model()