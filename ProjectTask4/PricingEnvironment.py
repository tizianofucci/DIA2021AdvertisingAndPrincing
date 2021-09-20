from Environment import Environment
import numpy as np

"""
Environment used for sampling rewards, given the pulled arm (price).
    mu_new: Average new potential customers for the day.
    sigma_new: Stddev of new potential customers for the day.
    returns_coeff: Paremeter affecting the mean of the poisson distribution used for sampling number of returns.
    bid_offset: Parameter affecting single-cost-per-click discount.
    
"""
class PricingEnvironment(Environment):
    
    def __init__(self,n_arms,prices, prod_cost, probabilities, contexts_bid_offsets, mu_new,sigma_new,contexts_n_returns_coeff,features_matrix):
        super().__init__(n_arms,probabilities)
        self.prices = prices
        self.prod_cost = prod_cost
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.features_matrix = features_matrix
        self.classes = [Customer_class(n_arms, prices, prod_cost, probabilities[features_matrix[_][0]][features_matrix[_][1]],contexts_bid_offsets[features_matrix[_][0]][features_matrix[_][1]], mu_new[features_matrix[_][0]][features_matrix[_][1]],sigma_new[features_matrix[_][0]][features_matrix[_][1]],contexts_n_returns_coeff[features_matrix[_][0]][features_matrix[_][1]],features_matrix[_]) for _ in range(len(features_matrix))]
    """
    Collects all rewards from classes of customers
    """
    def round(self,pulled_arm,bid):
        classes_returns = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            classes_returns[i] = self.classes[i].round(pulled_arm[i],bid)
        return classes_returns


"""
Class that simulates the behaviour of all customers with the same features
"""
class Customer_class(Environment):
    def __init__(self,n_arms,prices,prod_cost,probabilities,bid_offset,mu_new,sigma_new,n_returns_coeff,features_vector):
        super().__init__(n_arms,probabilities)
        self.bid_offset = bid_offset
        self.prod_cost = prod_cost
        self.prices = prices
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.n_returns_coeff = n_returns_coeff
        self.features_vector = features_vector
    
    def round(self,pulled_arm,bid):
        
        new_customers = abs(round(np.random.normal(self.mu_new,self.sigma_new)))
        single_rewards = np.zeros(np.sum(new_customers))
        single_cost_per_click = np.zeros(np.sum(new_customers))

        for i in range (0, new_customers):
            customer = Customer(self.probabilities,self.features_vector)
            single_rewards[i] = customer.round_costumer(pulled_arm)
            single_cost_per_click[i] = bid - abs(np.random.normal(bid, 0.1))/self.bid_offset

        
        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson((self.n_returns_coeff/(2*((self.prices[pulled_arm])/10)+0.5)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * (self.prices[pulled_arm] - self.prod_cost)) - costs
        return money_reward
     
class  Customer():
    def __init__(self,first_buy_probabilities,feature_vector):
        self.first_buy_probabilities = first_buy_probabilities
        self.feature_vector = feature_vector
    def round_costumer(self,pulled_arm):
        reward = np.random.binomial(1, self.first_buy_probabilities[pulled_arm])
        return reward
