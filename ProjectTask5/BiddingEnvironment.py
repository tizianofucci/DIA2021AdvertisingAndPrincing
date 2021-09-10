import numpy as np


# conversion prob is known
# reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs
class BiddingEnvironment():
    
    def __init__(self, n_arms, price, bids, bid_modifiers, conv_rate, mu_new, sigma_new):
        #super().__init__(n_arms, probabilities)
        self.price = price
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.conv_rate = conv_rate
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]

    def round(self,pulled_arm):
        delta_customers = self.bid_modifiers[pulled_arm]**2
        new_customers = round(np.random.normal((self.mu_new + 5*delta_customers),self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)

        for i in range (0, new_customers):
            customer = Customer(self.conv_rate)
            single_rewards[i] = customer.round_costumer()
            single_cost_per_click[i] = self.bids[pulled_arm] - abs(np.random.normal(0, self.bids[pulled_arm]/10))

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = abs(round(np.random.normal(15 - 2*self.price,3)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * self.price) - costs
        return money_reward

class Customer():
    def __init__(self,conv_rate):
        self.conv_rate = conv_rate

    def round_costumer(self):
        reward = np.random.binomial(1, self.conv_rate)
        return reward
