import numpy as np
import matplotlib.pyplot as plt

def calculate_sigma(sigma, new_sigma, T):
    sigma *= (T-1)
    sigma += new_sigma
    sigma /= T
    return sigma

def check_equilibrium(sigma, player_1_utility_array, player_2_utility_array):
    k = sigma.shape[0]
    res = 0

    #player 1
    for i in range(k):
        for l in range(k):
            temp = 0
            for j in range(k):
                temp += sigma[i][j] * (player_1_utility_array[l][j]-player_1_utility_array[i][j])
            res = max(temp, res)

    #player 2
    for j in range(k):
        for l in range(k):
            temp = 0
            for i in range(k):
                temp += sigma[i][j] * (player_2_utility_array[l][i]-player_2_utility_array[j][i])
            res = max(temp, res)

    return res

def find_stationary_distribution(transition_matrix):
    n = transition_matrix.shape[0]
    # Create an initial probability distribution
    distribution = np.ones(n) / n
    prev_distribution = np.zeros(n)
    
    # Iterate until convergence
    while np.linalg.norm(distribution - prev_distribution) > 1e-8:
        prev_distribution = distribution
        distribution = np.dot(distribution, transition_matrix)
    return distribution

def define_epsilon(n,T,a=1):
        return np.sqrt(np.log(n)/T)*a

class onlineHedgeInstance(object):

    def __init__(self, n=6, v=1):
        self.n = n
        self.actual_value = v
        self.possible_bids = np.arange(0, 1.1, 0.2)
        self.weights = np.ones(self.n)
        zeros = 0
        for i in range(n):
            if (self.possible_bids[i] > self.actual_value):
                zeros += 1
                self.weights[i] = 0
        self.weights /= (self.n-zeros)

class onlineHedge(object):

    def __init__(self, n=6, T=100000, v=1, a=1, no_swap=False):
        self.n = n
        self.T = T
        self.actual_value = v
        self.possible_bids = np.arange(0, 1.1, 0.2)
        self.epsilon = define_epsilon(self.n, self.T, a=a)
        self.no_swap = no_swap
        if not(no_swap):
            self.weights = np.ones(self.n)
            zeros = 0
            for i in range(n):
                if (self.possible_bids[i] > self.actual_value):
                    zeros += 1
                    self.weights[i] = 0
            self.weights /= (self.n-zeros)       
        else:
            self.consensus_distribution = [0] * self.n
            self.hedge_instances = np.array([onlineHedgeInstance(n=self.n, v=self.actual_value) for _ in range(self.n)])
    
    def utility(self, opponent_bid):
        utility_array = np.zeros(self.n)
        for i in range(self.n):
            if (round(self.possible_bids[i],1) > opponent_bid):
                utility_array[i] = self.actual_value - self.possible_bids[i]
            elif (round(self.possible_bids[i],1) == opponent_bid):  
                utility_array[i] = (self.actual_value - self.possible_bids[i])/2
        return utility_array

    def recalculate_weights(self, opponent_bid):
        #Calculating utilities
        utilities = self.utility(opponent_bid)
        if not(self.no_swap):
            self.weights*=np.exp(self.epsilon*utilities)
            self.weights/=np.sum(self.weights)
        else:
            for i in range(len(self.hedge_instances)):
                instance_utilities = utilities * self.consensus_distribution[i]
                self.hedge_instances[i].weights*=np.exp(self.epsilon*instance_utilities)
                self.hedge_instances[i].weights/=np.sum(self.hedge_instances[i].weights)

    def predict(self):
        if not(self.no_swap):
            return np.random.choice(self.possible_bids, p=self.weights)
        else:
            distributions = []
            for i in range(len(self.hedge_instances)):
                distributions.append(self.hedge_instances[i].weights)
            transition_matrix = np.array(distributions)
            self.consensus_distribution = find_stationary_distribution(transition_matrix)
            return np.random.choice(self.possible_bids, p=self.consensus_distribution)

def moving_average(a, l=3):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():
    epsilon_values = []
    revenue = 0
    no_swap_regret_algorith = True
    player_1 = onlineHedge(v=1, no_swap=no_swap_regret_algorith)
    player_2 = onlineHedge(v=1, no_swap=no_swap_regret_algorith)
    player_1_bids = []
    player_2_bids = []
    equilibrium = []
    sigma = np.zeros((player_1.n, player_1.n))
    player_1_utility_array = np.zeros((player_1.n, player_1.n))
    for i in range(player_1.n):
        for j in range(i):
            player_1_utility_array[i][j] = player_1.actual_value - i/5 
        player_1_utility_array[i][i] = (player_1.actual_value - i/5)/2 

    player_2_utility_array = np.zeros((player_1.n, player_1.n))
    for i in range(player_1.n):
        for j in range(i):
            player_2_utility_array[i][j] = player_2.actual_value - i/5 
        player_2_utility_array[i][i] = (player_2.actual_value - i/5)/2 

    for i in range(player_1.T):
        bid_1 = player_1.predict()
        bid_2 = player_2.predict()
        bid_1 = round(bid_1, 1)
        bid_2 = round(bid_2, 1)
        player_1_bids.append(bid_1)
        player_2_bids.append(bid_2)

        if (bid_1 > bid_2):
            revenue += bid_1
        else:
            revenue += bid_2   
        player_1.recalculate_weights(bid_2)
        player_2.recalculate_weights(bid_1)
        if (no_swap_regret_algorith):
            new_sigma = np.outer(player_1.consensus_distribution, player_2.consensus_distribution)
        else:
            new_sigma = np.outer(player_1.weights, player_2.weights)

        sigma = calculate_sigma(sigma, new_sigma, i+1)
        equilibrium.append(check_equilibrium(sigma, player_1_utility_array, player_2_utility_array))  
        epsilon_values.append(define_epsilon(n=6, T=i+1)) 

    print(f'The average revenue is', round(revenue/player_1.T, 3))
    print(f'epsilon is', player_1.epsilon)

    plt.figure()
    plt.plot(equilibrium, label = 'Equilibrium Over Time')
    plt.plot(epsilon_values, label = 'Epsilon Over Time [eps = sqrt(lnn/T)]')
    plt.ylabel("Equilibrium values")
    plt.xlabel("Time (number of auctions)")
    plt.legend()
    plt.show()
    
    mov_aver_1 = moving_average(player_1_bids, l=100)
    mov_aver_2 = moving_average(player_2_bids, l=100)
    plt.figure() 
    plt.plot(player_1_bids, label='Player 1', color='blue', linewidth=0.08)
    plt.plot(player_2_bids, label='Player 2', color='orange', linewidth=0.08)
    
    plt.plot(mov_aver_1, label = 'Moving average of player 1', color='black', linewidth=1)
    plt.plot(mov_aver_2, label = 'Moving average of player 2', color='red', linewidth=0.5)
    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("Bidding over time")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    main()    