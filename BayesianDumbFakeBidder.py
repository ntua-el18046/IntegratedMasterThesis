import numpy as np
import matplotlib.pyplot as plt

def define_epsilon(n,T,a=1):
        return np.sqrt(np.log(n)/T)*a

class onlineHedge(object):

    def __init__(self, n=101, T=1000000, v=1, a=1, faker=False):
        self.n = n
        self.T = T
        self.actual_value = v
        self.is_fake = faker
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
                utility_array[i] = self.actual_value - self.possible_bids[i]
            elif (self.possible_bids[i] == opponent_bid):  
                utility_array[i] = (self.actual_value - self.possible_bids[i])/2
        return utility_array

    def utility_of_fake_bidder(self, opponent_bid):
        utility_array = np.zeros(self.n)
        for i in range(self.n):
            if (self.possible_bids[i] <= opponent_bid):
                utility_array[i] = opponent_bid
        return utility_array

    def recalculate_weights(self, opponent_bid):
        if not(self.is_fake):
            utilities = self.utility(opponent_bid)
        else: 
            utilities = self.utility_of_fake_bidder(opponent_bid)  
        reweighting = np.exp(self.epsilon*utilities)
        self.weights*=reweighting
        self.weights/=np.sum(self.weights)

    def predict(self):
        return np.random.choice(self.possible_bids, p=self.weights)

def moving_average(a, l=100):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():
    revenue = 0
    num_of_players = 2 #without counting the fake bidder
    v1 = 0.2
    v2 = 0.8
    players = []
    bids = []
    
    for _ in range(num_of_players):
        players.append(onlineHedge(v=v2))
        players.append(onlineHedge(v=v1))
        bids.append([])
        bids.append([])

    # for the fake bidder        
    players.append(onlineHedge(v=1, faker=True))
    bids.append([])
    for i in range(players[0].T):
        current_players = []
        current_bids = []
        for j in range(0, len(players)-1, 2):
            res = np.random.choice([0, 1], p=[1/2, 1/2])
            if res == 0:
                current_players.append(players[j+1])
                current_bids.append(bids[j+1])
            else:  
                current_players.append(players[j])
                current_bids.append(bids[j])      
        
        current_players.append(players[-1]) 
        current_bids.append(bids[-1])
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
        
        # If the winner is not the fake bidder
        if (winner != len(current_players)-1 or 
            second_winner_value == winner_value):
            revenue += bid_values[winner]
        
        for i in range(num_of_players+1):
            if (bid_values[i] != winner_value):
                current_players[i].recalculate_weights(winner_value)
            else:
                current_players[i].recalculate_weights(second_winner_value)

    print(f'The average revenue is', round(revenue/(players[0].T), 3))
    
    printable_bids = []
    
    for i in range(len(players)-1):
        printable_bids.append([item for item in bids[i] for _ in range(2)])
    printable_bids.append(bids[-1])
    
    moving_averages = []
    for i in range(len(players)):
        moving_averages.append(moving_average(printable_bids[i]))
    
    colors = ['orange', 'brown', 'yellow', 'purple', 'olive', 'blue', 'purple']
    for i in range(0, len(players)-1, 2):
        plt.plot(printable_bids[i], label=f'Player {int(i/2)+1} - Valuation {v2}', color=colors[i], linewidth=0.1)
        plt.plot(printable_bids[i+1], label=f'Player {int(i/2)+1} - Valuation {v1}', color=colors[i+1], linewidth=0.1)
    plt.plot(printable_bids[-1], label=f'Fake Player', color=colors[-1], linewidth=0.1)
    
    aver_colors = ['red', 'red', 'green', 'black', 'brown', 'purple', 'pink']
    for i in range(0, len(players)-1, 2):
        plt.plot(moving_averages[i], label = f'Moving average of Player {int(i/2)+1} - Valuation {v2}', color=aver_colors[i], linewidth=1)    
        plt.plot(moving_averages[i+1], label = f'Moving average of Player {int(i/2)+1} - Valuation {v1}', color=aver_colors[i+1], linewidth=1)   
    plt.plot(moving_averages[-1], label = f'Moving average of Fake Player', color=aver_colors[-1], linewidth=1)
    
    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("Bidding over time")
    plt.legend(fontsize='xx-small')
    plt.show()
     

if __name__ == "__main__":
    main()

    