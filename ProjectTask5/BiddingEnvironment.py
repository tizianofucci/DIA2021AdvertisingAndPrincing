import numpy as np


# conversion prob is known
# reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs
class BiddingEnvironment():
    
    def __init__(self, n_arms, price, prod_cost, bids, bid_modifiers, conv_rate, mu_new, sigma_new, returns_coeffs, bid_offsets):
        #super().__init__(n_arms, probabilities)
        self.price = price
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.conv_rate = conv_rate
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.returns_coeffs = returns_coeffs
        self.bid_offsets = bid_offsets
        #self.results = ""

    def round(self,pulled_arm):
        delta_customers = 200*(self.bid_modifiers[pulled_arm]*2)
        new_customers = round(np.random.normal((self.mu_new + delta_customers),self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)

        for i in range (0, new_customers):
            customer = Customer(self.conv_rate)
            single_rewards[i] = customer.round_costumer()
            single_cost_per_click[i] = self.bids[pulled_arm] - abs(np.random.normal(self.bids[pulled_arm], 0.1))/self.bid_offsets[pulled_arm]

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            #n_returns[i] = max(0, (round(np.random.normal(15 - 2*self.price,1))))
            n_returns[i] = np.random.poisson((self.returns_coeffs[pulled_arm]/(2*((self.price)/10)+0.5)))
            #n_returns[i] = 15-2*self.price

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * (self.price - self.prod_cost)) - costs
        
        #self.results = "arm: {}, sales: {}, total_returns: {}, costs: {}, total:{}".format(pulled_arm, sells, total_returns, costs, money_reward)

        return money_reward

class Customer():
    def __init__(self,conv_rate):
        self.conv_rate = conv_rate

    def round_costumer(self):
        reward = np.random.binomial(1, self.conv_rate)
        return reward
