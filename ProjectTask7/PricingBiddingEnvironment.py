import numpy as np
from Environment import Environment


# conversion prob is known
# reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs
class PricingBiddingEnvironment():
    
    def __init__(self, prices, prod_cost, bids, bid_modifiers, contexts_bid_offsets, first_buy_probabilities, mu_new, sigma_new,delta_customers_multipliers, contexts_n_returns_coeffs,features_matrix):
        #super().__init__(n_arms, probabilities)
        self.idx_price = 1
        self.idx_bid = 0
        self.contexts_bid_offsets = contexts_bid_offsets
        self.prices = prices
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.first_buy_probabilities = first_buy_probabilities
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.delta_customers_multipliers = delta_customers_multipliers
        self.contexts_n_returns_coeffs = contexts_n_returns_coeffs
        self.features_matrix = features_matrix
        self.classes = [Customer_class(prices,prod_cost,bids,bid_modifiers[features_matrix[_][0]][features_matrix[_][1]],contexts_bid_offsets[features_matrix[_][0]][features_matrix[_][1]] , first_buy_probabilities[features_matrix[_][0]][features_matrix[_][1]],mu_new[features_matrix[_][0]][features_matrix[_][1]],sigma_new[features_matrix[_][0]][features_matrix[_][1]], delta_customers_multipliers[features_matrix[_][0]][features_matrix[_][1]], contexts_n_returns_coeffs[features_matrix[_][0]][features_matrix[_][1]],features_matrix[_]) for _ in range(len(features_matrix))]

        #self.results = ""

    def round(self,pulled_arm):
        classes_returns = np.zeros(len(self.classes))
        classes_number= np.zeros(len(self.classes))

        for i in range(len(self.classes)):
            classes_returns[i],classes_number[i] = self.classes[i].round(pulled_arm[i])
        return classes_returns,classes_number

class Customer_class():
    def __init__(self, prices, prod_cost, bids, bid_modifiers, bid_offset, probabilities, mu_new, sigma_new, delta_customers_multiplier, n_returns_coeff,features_vector):
        self.idx_price = 1
        self.idx_bid = 0
        self.delta_customers_multiplier = delta_customers_multiplier
        self.prices = prices
        self.prod_cost = prod_cost
        self.bids = bids
        self.bid_modifiers = bid_modifiers
        self.bid_offset = bid_offset
        self.probabilities = probabilities
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.n_returns_coeff = n_returns_coeff
        self.features_vector = features_vector
    
    #TODO differenziare le poisson per classe
    def round(self,pulled_arm):
        delta_customers = 200*(self.bid_modifiers[pulled_arm[self.idx_bid]])*self.delta_customers_multiplier
        new_customers = round(np.random.normal((self.mu_new + delta_customers),self.sigma_new))
        single_rewards = np.zeros(np.sum(new_customers))
        single_cost_per_click = np.zeros(np.sum(new_customers))

        for i in range (0, new_customers):
            customer = Customer(self.probabilities)
            single_rewards[i] = customer.round_costumer(pulled_arm[self.idx_price])
            single_cost_per_click[i] = self.bids[pulled_arm[self.idx_bid]] - abs(np.random.normal(self.bids[pulled_arm[self.idx_bid]], 0.1))/self.bid_offset

        
        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson((self.n_returns_coeff/(2*((self.prices[pulled_arm[self.idx_price]])/10)+0.5)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        money_reward = ((sells + total_returns) * (self.prices[pulled_arm[self.idx_price]] - self.prod_cost)) - costs
        return money_reward,new_customers



class Customer():
    def __init__(self,first_buy_probabilities):
        self.first_buy_probabilities = first_buy_probabilities
    def round_costumer(self,idx):
        reward = np.random.binomial(1, self.first_buy_probabilities[idx])
        return reward
