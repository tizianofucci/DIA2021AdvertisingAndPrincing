from numpy.core.fromnumeric import prod
from Environment import Environment
import numpy as np

class PricingEnvironment(Environment):
    
    def __init__(self,n_arms,prices,prod_cost,probabilities,mu_new,sigma_new,returns_coeff, bid_offset):
        super().__init__(n_arms,probabilities)
        self.prices = prices
        self.prod_cost = prod_cost
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.returns_coeff = returns_coeff
        self.bid_offset = bid_offset
        self.total_returns_per_arm = [[] for _ in range(n_arms)]

    def round(self,pulled_arm,bid):
        new_customers = round(np.random.normal(self.mu_new,self.sigma_new))
        single_rewards = np.zeros(new_customers)
        single_cost_per_click = np.zeros(new_customers)

        for i in range (0, new_customers):
            customer = Customer(self.probabilities)
            single_rewards[i] = customer.round_costumer(pulled_arm)
            single_cost_per_click[i] = bid - abs(np.random.normal(bid, 0.1))/self.bid_offset

        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson((self.returns_coeff/(2*((self.prices[pulled_arm])/10)+0.5)))

        total_returns = np.sum(n_returns)
        costs = np.sum(single_cost_per_click)
        production_costs = self.prod_cost*(total_returns + sells)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs - production_costs
        return money_reward

        
class  Customer():
    def __init__(self,conv_rates):
        self.conv_rates = conv_rates
    def round_costumer(self,pulled_arm):
        reward = np.random.binomial(1, self.conv_rates[pulled_arm])
        return reward
