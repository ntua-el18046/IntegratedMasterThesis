import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, n=101, v=1):
        self.n = n
        self.actual_value = v
        self.possible_bids = np.arange(0, self.n/(self.n-1), 1/(self.n-1))
        self.weights = np.ones(self.n)
        zeros = 0
        for i in range(n):
            if (self.possible_bids[i] > self.actual_value):
                zeros += 1
                self.weights[i] = 0
        self.weights /= (self.n-zeros)

class onlineHedge(object):

    def __init__(self, n=101, T=100000, v=1, a=1):
        self.n = n
        self.T = T
        self.actual_value = v
        self.possible_bids = np.arange(0, self.n/(self.n-1), 1/(self.n-1))
        self.consensus_distribution = [0] * self.n
        self.epsilon = define_epsilon(self.n, self.T, a=a)
        self.hedge_instances = np.array([onlineHedgeInstance(n=self.n, v=self.actual_value) for _ in range(self.n)])

    def utility(self, opponent_bid):
        utility_array = np.zeros(self.n)
        for i in range(self.n):
            if (self.possible_bids[i] > opponent_bid):
                utility_array[i] = self.actual_value - self.possible_bids[i]
            elif (self.possible_bids[i] == opponent_bid):  
                utility_array[i] = (self.actual_value - self.possible_bids[i])/2
        return utility_array

    def recalculate_weights(self, opponent_bid):
        utilities = self.utility(opponent_bid) 
        for i in range(len(self.hedge_instances)):
            instance_utilities = utilities * self.consensus_distribution[i]
            self.hedge_instances[i].weights*=np.exp(self.epsilon*instance_utilities)
            self.hedge_instances[i].weights/=np.sum(self.hedge_instances[i].weights)
    
    def predict(self):
        distributions = []
        for i in range(len(self.hedge_instances)):
            distributions.append(self.hedge_instances[i].weights)
        transition_matrix = np.array(distributions)
        self.consensus_distribution = find_stationary_distribution(transition_matrix)
        return np.random.choice(self.possible_bids, p=self.consensus_distribution)

def moving_average(a, l=100):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():

    revenue = 0
    num_of_players = 2 
    v1 = 0.5
    v2 = 1
    players = []
    bids = []
    
    for _ in range(num_of_players):
        players.append(onlineHedge(v=v2))
        players.append(onlineHedge(v=v1))
        bids.append([])
        bids.append([])

    for i in range(players[0].T):
        current_players = []
        current_bids = []
        for j in range(0, len(players), 2):
            res = np.random.choice([0, 1], p=[1/2, 1/2])
            if res == 0:
                current_players.append(players[j+1])
                current_bids.append(bids[j+1])
            else:  
                current_players.append(players[j])
                current_bids.append(bids[j])      
        
        bid_values = []
        for i in range(len(current_players)):
            x = current_players[i].predict()
            bid_values.append(x)
            current_bids[i].append(x)

        #Sort the bids for comparison
        # Create a list of tuples (index, value)
        indexed_bid_values = list(enumerate(bid_values))

        # Sort the list of tuples based on the value in descending order
        sorted_bid_values = sorted(indexed_bid_values, key=lambda pair: pair[1], reverse=True)

        # Extract the indices from the sorted list of tuples
        winner, winner_value = sorted_bid_values[0]
        _, second_winner_value = sorted_bid_values[1]
        
        revenue += bid_values[winner]  
        for i in range(num_of_players):
            if (bid_values[i] != winner_value):
                current_players[i].recalculate_weights(winner_value)
            else:
                current_players[i].recalculate_weights(second_winner_value)

    print(f'The average revenue is', round(revenue/(players[0].T), 3))    
    moving_averages = []
    for i in range(len(players)):
        moving_averages.append(moving_average(bids[i]))
    
    colors = ['purple', 'olive', 'orange', 'blue', 'yellow', 'red', 'purple']
    for i in range(0, len(players)-1, 2):
        plt.plot(bids[i], label=f'Player {int(i/2)+1} - Valuation {v2}', color=colors[i], linewidth=0.1)
        plt.plot(bids[i+1], label=f'Player {int(i/2)+1} - Valuation {v1}', color=colors[i+1], linewidth=0.1)
    
    aver_colors = ['black', 'green', 'red', 'cyan', 'purple', 'brown', 'pink']
    for i in range(0, len(players)-1, 2):
        plt.plot(moving_averages[i], label = f'Moving average of Player {int(i/2)+1} - Valuation {v2}', color=aver_colors[i], linewidth=1)    
        plt.plot(moving_averages[i+1], label = f'Moving average of Player {int(i/2)+1} - Valuation {v1}', color=aver_colors[i+1], linewidth=1)   
    
    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("Bidding over time")
    plt.legend(fontsize='xx-small')
    plt.show()

if __name__ == "__main__":
    main()