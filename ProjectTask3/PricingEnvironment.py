from Environment import Environment
import numpy as np

class PricingEnvironment(Environment):
    
    def __init__(self,n_arms,prices,probabilities,mu_new,sigma_new):
        super().__init__(n_arms,probabilities)
        self.prices = prices
        self.mu_new = mu_new
        self.sigma_new = sigma_new

    def round(self,pulled_arm):
        new_customers = round(np.random.normal(self.mu_new,self.sigma_new))
        single_rewards = np.zeros(new_customers)
        for i in range (0, new_customers):
            customer = Customer(self.probabilities)
            single_rewards[i] = customer.round_costumer(pulled_arm)
        sells = np.sum(single_rewards)
        money_reward = sells * self.prices[pulled_arm]
        return money_reward

        
class  Customer():
    def __init__(self,first_buy_probabilities):
        self.first_buy_probabilities = first_buy_probabilities
    def round_costumer(self,pulled_arm):
        reward = np.random.binomial(1, self.first_buy_probabilities[pulled_arm])
        return reward
