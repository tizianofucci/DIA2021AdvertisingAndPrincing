import numpy as np


"""
Environment used for sampling rewards, given the pulled arm (bid).
    bid_modifiers: Multipliers affecting the number of clicks based on the bid
    mu_new: baseline for new potential customers for the day(independent from bid).
    sigma_new: Stddev of new potential customers for the day.
    returns_coeff: Paremeter affecting the mean of the poisson distribution used for sampling number of returns.
    bid_offset: Parameter affecting single-cost-per-click discount.
    
"""
class BiddingEnvironment():
    
    def __init__(self, n_arms, price, prod_cost, bids, bid_modifiers, conv_rates, mu_new, sigma_new, returns_coeffs, bid_offsets):
        self.price = price
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.conv_rates = conv_rates
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.returns_coeffs = returns_coeffs
        self.bid_offsets = bid_offsets
        
    def round(self,pulled_arm):
        delta_customers = 200*(self.bid_modifiers[pulled_arm])
        new_customers = round(np.random.normal((self.mu_new + delta_customers),self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)

        for i in range (0, new_customers):
            customer = Customer(self.conv_rates[pulled_arm])
            single_rewards[i] = customer.round_costumer()
            single_cost_per_click[i] = self.bids[pulled_arm] - abs(np.random.normal(self.bids[pulled_arm], 0.1))/self.bid_offsets[pulled_arm]

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson((self.returns_coeffs[pulled_arm]/(2*((self.price)/10)+0.5)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * (self.price - self.prod_cost)) - costs
        return money_reward

class Customer():
    def __init__(self,conv_rate):
        self.conv_rate = conv_rate

    def round_costumer(self):
        reward = np.random.binomial(1, self.conv_rate)
        return reward
