import numpy as np
import matplotlib.pyplot as plt

def define_epsilon(n,T,a=1):
        return np.sqrt(np.log(n)/T)*a

class onlineHedge(object):

    def __init__(self, n=101, T=1000000, v=1, a=1):
        self.n = n
        self.T = T
        self.actual_value = v
        self.possible_bids = np.arange(0, self.n/(self.n-1), 1/(self.n-1))
        self.weights = np.ones(self.n)
        zeros = 0
        for i in range(n):
            if (self.possible_bids[i] > self.actual_value):
                zeros += 1
                self.weights[i] = 0
        self.weights /= (self.n-zeros)
        self.epsilon = define_epsilon(self.n, self.T, a=a)


    def utility(self, opponent_bid):
        utility_array = np.zeros(self.n)
        for i in range(self.n):
            if (self.possible_bids[i] > opponent_bid):
                utility_array[i] = (self.actual_value - self.possible_bids[i])
            elif (self.possible_bids[i] == opponent_bid):  
                utility_array[i] = (self.actual_value - self.possible_bids[i])/2
        return utility_array


    def recalculate_weights(self, opponent_bid):
        utilities = self.utility(opponent_bid)
        reweighting = np.exp(self.epsilon*utilities)
        self.weights*=reweighting
        self.weights/=np.sum(self.weights)

    def predict(self):
        return np.random.choice(self.possible_bids, p=self.weights)

def moving_average(a, l):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():
    revenue = 0
    revenue_list = []
    num_of_players = 2 
    v1 = 0.5
    v2 = 1
    players = []
    bids = []
    
    for _ in range(num_of_players):
        players.append(onlineHedge(v=v1))
        players.append(onlineHedge(v=v2))
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
        revenue_list.append(bid_values[winner])  
        for i in range(num_of_players):
            if (bid_values[i] != winner_value):
                current_players[i].recalculate_weights(winner_value)
            else:
                current_players[i].recalculate_weights(second_winner_value)

    print(f'The average revenue is', round(revenue/(players[0].T), 4))

    mov_aver_revenue = moving_average(revenue_list, l=1000)
    
    plt.plot(mov_aver_revenue, label = 'Moving average of revenue', color='purple', linewidth=1)
    plt.ylabel('Revenue per transaction')
    plt.xlabel('Time (number of auctions)')
    plt.title('Revenue over time')
    plt.show()


if __name__ == "__main__":
    main()

    