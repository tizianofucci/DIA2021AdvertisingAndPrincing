import numpy as np


"""
Environment used for sampling rewards, given the pulled arm (pair: bid,price).
    bid_modifiers: Multipliers affecting the number of clicks based on the bid
    mu_new: baseline for new potential customers for the day(independent from bid).
    sigma_new: Stddev of new potential customers for the day.
    returns_coeff: Paremeter affecting the mean of the poisson distribution used for sampling number of returns.
    bid_offset: Parameter affecting single-cost-per-click discount.
    
"""
class PricingBiddingEnvironment():
    
    def __init__(self, prices, prod_cost, bids, bid_modifiers, first_buy_probabilities, mu_new, sigma_new,  returns_coeffs, bid_offsets):
        self.idx_price = 1
        self.idx_bid = 0

        self.prices = prices
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.first_buy_probabilities = first_buy_probabilities
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.returns_coeffs = returns_coeffs
        self.bid_offsets = bid_offsets
        

    def round(self,pulled_arm):
        delta_customers = 200*(self.bid_modifiers[pulled_arm[self.idx_bid]])
        new_customers = round(np.random.normal((self.mu_new + delta_customers),self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)
        returns_coeff = self.returns_coeffs[pulled_arm[self.idx_bid]]
        bid_offset = self.bid_offsets[pulled_arm[self.idx_bid]]

        for i in range (0, new_customers):
            customer = Customer(self.first_buy_probabilities[pulled_arm[self.idx_bid]])
            single_rewards[i] = customer.round_costumer(pulled_arm[self.idx_price])
            single_cost_per_click[i] = self.bids[pulled_arm[self.idx_bid]] - abs(np.random.normal(self.bids[pulled_arm[self.idx_bid]], 0.1))/bid_offset

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))


        
        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson((returns_coeff/(2*((self.prices[pulled_arm[self.idx_price]])/10)+0.5)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        
        money_reward = ((sells + total_returns) * (self.prices[pulled_arm[self.idx_price]] - self.prod_cost)) - costs
        

        return money_reward

class Customer():
    def __init__(self,first_buy_probabilities):
        self.first_buy_probabilities = first_buy_probabilities
    def round_costumer(self,idx):
        reward = np.random.binomial(1, self.first_buy_probabilities[idx])
        return reward
