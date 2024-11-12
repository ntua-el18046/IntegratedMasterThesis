import numpy as np
import matplotlib.pyplot as plt

def define_epsilon(n,T,a=1):
        return np.sqrt(np.log(n)/T)*a

class onlineHedge(object):

    def __init__(self, n=101, T=10000, v=1, a=1):
        self.n = n
        self.T = T
        self.actual_value = v 
        #The possible bids correspond to the experts' predictions
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
            if (self.possible_bids[i] < opponent_bid):
                utility_array[i] = opponent_bid
            elif (self.possible_bids[i] == opponent_bid):  
                utility_array[i] = opponent_bid/2
        return utility_array


    def recalculate_weights(self, opponent_bid, fake_bidder=False):
        if not(fake_bidder):
            utilities = self.utility(opponent_bid)
        else: 
            utilities = self.utility_of_fake_bidder(opponent_bid)  

        #Apply weights
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
    v1 = 1
    v2 = 0
    player_1 = onlineHedge(v=v1)
    player_2 = onlineHedge(v=v2)
    player_3 = onlineHedge(v=1) # fake bidder
    player_1_bids = []
    player_2_bids = []
    player_3_bids = []
    for i in range(player_1.T):
        bid_1 = player_1.predict()
        bid_2 = player_2.predict()
        bid_3 = player_3.predict()
        player_1_bids.append(bid_1)
        player_2_bids.append(bid_2)
        player_3_bids.append(bid_3)

        greater = max(bid_1, bid_2)
        if (bid_3 > greater):
            player_1.recalculate_weights(bid_3)
            player_2.recalculate_weights(bid_3)
            if (greater > bid_1):
                player_3.recalculate_weights(bid_2, fake_bidder=True)
            else:
                player_3.recalculate_weights(bid_1, fake_bidder=True)
        elif (bid_1 > bid_2):
            revenue += bid_1
            player_2.recalculate_weights(bid_1)
            player_3.recalculate_weights(bid_1, fake_bidder=True)
            if bid_2 > bid_3:
                player_1.recalculate_weights(bid_2)
            else:
                player_1.recalculate_weights(bid_3)
        else:
            revenue += bid_2   
            player_1.recalculate_weights(bid_2)
            player_3.recalculate_weights(bid_2, fake_bidder=True)
            if bid_1 > bid_3:
                player_2.recalculate_weights(bid_1)
            else: 
                player_2.recalculate_weights(bid_3)
   

    print(f'The average revenue is', round(revenue/player_1.T, 3))
    mov_aver_1 = moving_average(player_1_bids)
    mov_aver_2 = moving_average(player_2_bids)
    mov_aver_3 = moving_average(player_3_bids)
    plt.plot(player_1_bids, label=f'Player 1 - Valuation {v1}', color='blue', linewidth=0.08)
    plt.plot(player_2_bids, label=f'Player 2 - Valuation {v2}', color='orange', linewidth=0.08)
    plt.plot(player_3_bids, label='Fake Bidder', color='pink', linewidth=0.08)
    
    plt.plot(mov_aver_1, label = 'Moving average of player 1', color='black', linewidth=1)
    plt.plot(mov_aver_2, label = 'Moving average of player 2', color='red', linewidth=0.5)
    plt.plot(mov_aver_3, label = 'Moving average of fake bidder', color='green', linewidth=0.5)
    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("Bidding over time")
    plt.legend()
    plt.show()

        

if __name__ == "__main__":
    main()

    