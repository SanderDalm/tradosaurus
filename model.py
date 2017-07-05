from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################
# Read stock data
###############################################

stocks = pd.read_csv('data/stocks.csv')

###############################################
# Create features for NN
###############################################

def create_features(df, n_hist):
    
    features = []
    labels = []
    
    for aandeel in df.Aandeel.unique().tolist():
        
        prices = np.array(df[df.Aandeel==aandeel].Close.tolist())  

        cursor = 0
        offset = 10
        features = []
        
        while cursor < len(prices):
            features.append(prices[cursor:cursor+n_hist])
            
        
        stukjes = len(prices)/100
        split_prices = np.split(prices, stukjes)
        for price in split_prices:
            
            features.append(price)
            labels.append(np.sum(price[-7:-1]))
    return features, labels

stocks = pd.read_csv('stocks.csv')

features_train, labels_train = create_features(stocks)
        

###############################################
# Tradosaurus
###############################################

from tradosaurus import tradosaurus

trader = tradosaurus(num_nodes_hidden=[30, 30], gen_size=100, learning_rate=.1, decay=.999,
                         num_features=93,  mean=0, sd=.01)
trader.evolve(100, features_train, features_test)

plt.plot(np.array(features_train).T)

weights = trader.return_weights()
actions, reward = trader.trade(features_test, weights)
actions = np.array(actions)
mean = np.mean(actions)
std = np.std(actions)
actions = (actions-mean)/std
ranking = np.array(actions).argsort(axis=0)


plt.scatter(actions, labels_test)

# Inspect high and low predictions
best6 = ranking[-6:]
worst6 = ranking[:6]

def plotjes(ranking):
    hoofdplot = plt.figure()
    for index, value in enumerate(ranking):
        value = value[0]
        print value
        subplot = hoofdplot.add_subplot(2,3, index+1)
        subplot.plot(tradosaurus_test[value])
    hoofdplot.subplots_adjust(wspace=2, hspace=2)
    hoofdplot.show()


plotjes(best6)
plotjes(worst6)

# Separation chart
bar_chart = plt.figure()
ax1 = bar_chart.add_subplot(111)

pos_stocks = np.where(labels_test>0)[0]
neg_stocks = np.where(labels_test<0)[0]
pos_stocks = actions[pos_stocks]
neg_stocks = actions[neg_stocks]

ax1.hist([pos_stocks,neg_stocks], bins=10, color=['g', 'r'], alpha=.6)
ax1.grid(True)
ax1.set_title('Mean action for good and bad stocks')
ax1.legend(['Good stocks', 'Bad stocks'])
bar_chart.show()
