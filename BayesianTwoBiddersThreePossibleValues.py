import numpy as np
import matplotlib.pyplot as plt

def define_epsilon(n,T,a=1):
        return np.sqrt(np.log(n)/T)*a

class onlineHedge(object):

    def __init__(self, n=101, T=10000, v=1, a=1):
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

def moving_average(a, l=100):
    ret = np.cumsum(a, dtype=float)
    ret[l:] = ret[l:] - ret[:-l]
    return ret[l - 1:] / l    

def main():
    revenue = 0
    v1 = 1
    v2 = 0.5
    v3 = 0

    for numbers in range(1, 3):
        globals()['player_%s_evaluation_1' % numbers] = onlineHedge(v=v1)
        globals()['player_%s_evaluation_0' % numbers] = onlineHedge(v=v2)
        globals()['player_%s_evaluation_half' % numbers] = onlineHedge(v=v3)
        globals()['player_%s_eval_1_bids' % numbers] = []
        globals()['player_%s_eval_0_bids' % numbers] = []
        globals()['player_%s_eval_half_bids' % numbers] = []
        

    for i in range(player_1_evaluation_0.T):
        res = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
        
        if res == 0:
            player_1 = player_1_evaluation_0
            player_1_bids = player_1_eval_0_bids
        elif res == 1:
            player_1 = player_1_evaluation_1
            player_1_bids = player_1_eval_1_bids   
        else:  
            player_1 = player_1_evaluation_half
            player_1_bids = player_1_eval_half_bids        

        res_2 = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
        if res_2 == 0:
            player_2 = player_2_evaluation_0
            player_2_bids = player_2_eval_0_bids
        elif res_2 == 1:
            player_2 = player_2_evaluation_1
            player_2_bids = player_2_eval_1_bids
        else: 
            player_2 = player_2_evaluation_half
            player_2_bids = player_2_eval_half_bids

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

    print(f'The average revenue is', round(revenue/(player_1.T), 3))
    
    mov_aver_1 = moving_average(player_1_eval_0_bids, l=100)
    mov_aver_2 = moving_average(player_1_eval_1_bids, l=100)
    mov_aver_3 = moving_average( player_2_eval_0_bids, l=100)
    mov_aver_4 = moving_average( player_2_eval_1_bids, l=100)
    mov_aver_5 = moving_average( player_1_eval_half_bids, l=100)
    mov_aver_6 = moving_average( player_2_eval_half_bids, l=100)

    plt.plot(player_1_eval_0_bids, label=f'Player 1 - Valuation {v1}', color='blue', linewidth=0.1)
    plt.plot(player_1_eval_1_bids, label=f'Player 1 - Valuation {v2}', color='olive', linewidth=0.1)
    plt.plot(player_1_eval_half_bids, label=f'Player 1 - Valuation {v3}', color='red', linewidth=0.1)
    plt.plot(player_2_eval_0_bids, label=f'Player 2 - Valuation {v1}', color='brown', linewidth=0.1)
    plt.plot(player_2_eval_1_bids, label=f'Player 2 - Valuation {v2}', color='orange', linewidth=0.1)
    plt.plot(player_2_eval_half_bids, label=f'Player 2 - Valuation {v3}', color='purple', linewidth=0.1)
    
    plt.plot(mov_aver_1, label=f'Moving average of Player 1 - Valuation {v1}', color='purple', linewidth=1)
    plt.plot(mov_aver_2, label=f'Moving average of Player 1 -  Valuation {v2}', color='red', linewidth=1)
    plt.plot(mov_aver_5, label=f'Moving average of Player 1 -  Valuation {v3}', color='brown', linewidth=1)
    plt.plot(mov_aver_3, label=f'Moving average of Player 2 -  Valuation {v1}', color='black', linewidth=1)
    plt.plot(mov_aver_4, label=f'Moving average of Player 2 -  Valuation {v2}', color='green', linewidth=1)
    plt.plot(mov_aver_6, label=f'Moving average of Player 2 -  Valuation {v3}', color='black', linewidth=1)

    plt.ylabel("Bid level")
    plt.xlabel("Time (number of auctions)")
    plt.title("Bayesian Frist Price Auction - Bidding over time")
    plt.legend(fontsize='xx-small')
    plt.show()


if __name__ == "__main__":
    main()

    