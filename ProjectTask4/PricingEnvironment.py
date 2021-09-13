from matplotlib.pyplot import new_figure_manager
from scipy.stats.morestats import _calc_uniform_order_statistic_medians
from Environment import Environment
import numpy as np

class PricingEnvironment(Environment):
    
    def __init__(self,n_arms,prices,probabilities,mu_new,sigma_new,features_matrix):
        super().__init__(n_arms,probabilities)
        self.prices = prices
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.features_matrix = features_matrix
        self.classes = [Customer_class(n_arms,prices,probabilities[features_matrix[_][0]][features_matrix[_][1]],mu_new[features_matrix[_][0]][features_matrix[_][1]],sigma_new[features_matrix[_][0]][features_matrix[_][1]],features_matrix[_]) for _ in range(len(features_matrix))]

    #TODO split reward per class
    def round(self,pulled_arm,bid):
        classes_returns = np.zeros(len(self.classes))
        classes_number= np.zeros(len(self.classes))

        for i in range(len(self.classes)):
            classes_returns[i],classes_number[i] = self.classes[i].round(pulled_arm[i],bid)
        return classes_returns,classes_number



class Customer_class(Environment):
    def __init__(self,n_arms,prices,probabilities,mu_new,sigma_new,features_vector):
        super().__init__(n_arms,probabilities)
        self.prices = prices
        self.mu_new = mu_new
        self.sigma_new = sigma_new
        self.total_returns_per_arm = [[] for _ in range(n_arms)]
        self.features_vector = features_vector
    
    #TODO differenziare le poisson per classe
    def round(self,pulled_arm,bid):
        
        new_customers = abs(round(np.random.normal(self.mu_new,self.sigma_new)))
        single_rewards = np.zeros(np.sum(new_customers))
        single_cost_per_click = np.zeros(np.sum(new_customers))

        for i in range (0, new_customers):
            customer = Customer(self.probabilities,self.features_vector)
            single_rewards[i] = customer.round_costumer(pulled_arm)
            single_cost_per_click[i] = bid - abs(np.random.normal(0,bid/10)) 

        
        sells = np.sum(single_rewards)
        n_returns = np.zeros(int(sells))

        for i in range(0,int(sells)):
            n_returns[i] = np.random.poisson(3.0/(2*(self.prices[pulled_arm] - 3.5)))

        costs = np.sum(single_cost_per_click)
        total_returns = np.sum(n_returns)
        self.total_returns_per_arm[pulled_arm] = np.append(self.total_returns_per_arm[pulled_arm],total_returns)
        money_reward = ((sells + total_returns) * self.prices[pulled_arm]) - costs
        return money_reward,new_customers
     
class  Customer():
    def __init__(self,first_buy_probabilities,feature_vector):
        self.first_buy_probabilities = first_buy_probabilities
        self.feature_vector = feature_vector
    def round_costumer(self,pulled_arm):
        reward = np.random.binomial(1, self.first_buy_probabilities[pulled_arm])
        return reward
