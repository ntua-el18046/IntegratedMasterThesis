import numpy as np
import matplotlib.pyplot as plt

def define_epsilon(n,T,a=1):
        """
        Calculates a factor that is used to determine loss in the hedge algorithm

        Args:
            n (int): number of different possible bits
            T (int): number of time steps taken
            a (float): value that we can use to scale our epsilon
        Returns:
            epsilon (float): the theoretical epsilon, which can be also customized by a
        """

        return np.sqrt(np.log(n)/T)*a

class onlineHedge(object):

    def __init__(self, n=101, T=10000, v=1, a=1):
        self.n = n
        self.T = T
        self.actual_value = v 
        #The possible bids correspond to the experts' predictions
        self.possible_bids = np.arange(0, self.n/(self.n-1), 1/(self.n-1))
        self.weights = np.ones(self.n)
        # Avoiding overbidding
        zeros = 0
        for i in range(n):
            if (self.possible_bids[i] > self.actual_value):
                zeros += 1
                self.weights[i] = 0
        self.weights /= (self.n-zeros)
        self.epsilon = define_epsilon(self.n, self.T, a=a)


    def utility(self, opponent_bid):
        """
        Returns utlity of each player,
        where utility is defined as actual_value - bid,
        if the player won and 0 otherwise
        """

        utility_array = np.zeros(self.n)
        for i in range(self.n):
            if (self.possible_bids[i] > opponent_bid):
                utility_array[i] = self.actual_value - self.possible_bids[i]
            elif (self.possible_bids[i] == opponent_bid):  
                utility_array[i] = (self.actual_value - self.possible_bids[i])/2
        return utility_array


    def recalculate_weights(self, opponent_bid):
        """
        Readjusts the weights for each of the bids based on the opponent's actual bid value
        """

        #Calculating losses
        utilities = self.utility(opponent_bid)

        #Apply weights
        reweighting = np.exp(self.epsilon*utilities)
        self.weights*=reweighting
        self.weights/=np.sum(self.weights)

    def predict(self):
        """
        Chooses one bid based on the weights that have been calculated 
        and assigned to each possible bid by the Hedge algorithm
        """

        return np.random.choice(self.possible_bids, p=self.weights)

#sliding window technique
def moving_average(a, l=100):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():
    revenue = 0
    v1 = 1
    v2 = 1
    player_1 = onlineHedge(v=v1)
    player_2 = onlineHedge(v=v2)
    player_1_bids = []
    player_2_bids = []
    for i in range(player_1.T):
        bid_1 = player_1.predict()
        bid_2 = player_2.predict()
        player_1_bids.append(bid_1)
        player_2_bids.append(bid_2)

        if (bid_1 > bid_2):
            revenue += bid_1
        else:
            revenue += bid_2   
        player_1.recalculate_weights(bid_2)
        player_2.recalculate_weights(bid_1)

    print(f'The average revenue is', round(revenue/player_1.T, 3))
    mov_aver_1 = moving_average(player_1_bids)
    mov_aver_2 = moving_average(player_2_bids)
    plt.plot(player_1_bids, label=f'Player 1  - Valuation {v1}', color='purple', linewidth=0.08)
    plt.plot(player_2_bids, label=f'Player 2 - Valuation {v2}', color='orange', linewidth=0.08)
    plt.plot(mov_aver_1, label = 'Moving average of player 1', color='black', linewidth=1)
    plt.plot(mov_aver_2, label = 'Moving average of player 2', color='red', linewidth=0.5)
    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("First Price Auction - Bidding over time")
    plt.legend()
    plt.show()

        

if __name__ == "__main__":
    main()

    