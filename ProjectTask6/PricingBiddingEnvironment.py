import numpy as np


# conversion prob is known
# reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs
class PricingBiddingEnvironment():
    
    def __init__(self, prices, prod_cost, bids, bid_modifiers, first_buy_probabilities, mu_new, sigma_new):
        #super().__init__(n_arms, probabilities)
        self.idx_price = 1
        self.idx_bid = 0

        self.prices = prices
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.first_buy_probabilities = first_buy_probabilities
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        
        #self.results = ""

    def round(self,pulled_arm):
        delta_customers = 200*(self.bid_modifiers[pulled_arm[self.idx_bid]]*2)
        new_customers = round(np.random.normal((self.mu_new + delta_customers),self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)

        for i in range (0, new_customers):
            customer = Customer(self.conv_rate)
            single_rewards[i] = customer.round_costumer(pulled_arm[self.idx_price])
            single_cost_per_click[i] = self.bids[pulled_arm[self.idx_bids]] - abs(np.random.normal(self.bids[pulled_arm[self.idx_bid]], 0.1))/10

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            #n_returns[i] = max(0, (round(np.random.normal(15 - 2*self.price,1))))
            n_returns[i] = np.random.poisson(3.0/(2*(self.price - 3.5)))
            #n_returns[i] = 15-2*self.price

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        
        money_reward = ((sells + total_returns) * (self.price - self.prod_cost)) - costs
        
        #self.results = "arm: {}, sales: {}, total_returns: {}, costs: {}, total:{}".format(pulled_arm, sells, total_returns, costs, money_reward)

        return money_reward

class Customer():
    def __init__(self,first_buy_probabilities):
        self.first_buy_probabilities = first_buy_probabilities
    def round_costumer(self,idx):
        reward = np.random.binomial(1, self.first_buy_probabilities[idx])
        return reward
