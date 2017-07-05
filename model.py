from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression

###############################################
# Read stock data
###############################################

stocks = pd.read_csv('data/stocks.csv', usecols=['Date', 'Aandeel', 'Close'])

stocks = StockDataFrame.retype(stocks)
_ = stocks['macd']
      
stock_list = np.array(stocks.aandeel.unique().tolist())

np.random.shuffle(stock_list)
train_stocks = stock_list[:446]
test_stocks = stock_list[446:]

stocks_train = stocks[stocks.aandeel.isin(train_stocks)]
stocks_test = stocks[stocks.aandeel.isin(test_stocks)]

stocks_train['date'] = stocks_train.index
stocks_test['date'] = stocks_test.index
           
###############################################
# Create features for filtering
###############################################

def create_filter_features(df, n_future):
    
    df.sort_values(['aandeel','date'])
    
    features = []
    labels = []
    
    for aandeel in tqdm(df.aandeel.unique().tolist()):
                        
        prices = np.array(df[df.aandeel==aandeel].close.tolist())          
        macd = np.array(df[df.aandeel==aandeel].macdh.tolist())  
        ema12 = np.array(df[df.aandeel==aandeel].close_12_ema.tolist())  
        ema26 = np.array(df[df.aandeel==aandeel].close_26_ema.tolist())  
        
        cursor = 2
        offset = 10
        
        while cursor < len(prices)-n_future:
                        
            # Create index features from ema
            macd_change = macd[cursor]-macd[cursor-1]            
            #if macd[cursor]>macd[cursor-1]:
            #    macd_change=1
            #else:
            #    macd_change=0
            
            ema_ratio = ema12[cursor]/ema26[cursor]
            
            #if ema12[cursor]>ema26[cursor]:
            #    ema_ratio=1
            #else:
            #    ema_ratio=0

            features.append([macd_change, ema_ratio])
            
            # Calculate price change
            label = prices[cursor+n_future]/prices[cursor]
            #if prices[cursor+n_future]/prices[cursor]>1:
            #    label=1
            #else: label=0
                
            labels.append(label)
            cursor += offset
            
    return np.array(features), np.array(labels)
       

###############################################
# Fit models
###############################################     

features_train, labels_train = create_filter_features(stocks_train, 10)
features_test, labels_test = create_filter_features(stocks_test, 10)

model = LinearRegression()
model.fit(features_train, labels_train)
model.score(features_test, labels_test)
plt.scatter(model.predict(features_test), labels_test)

#temp = np.concatenate([features_train, labels_train.reshape([len(labels_train),1])], axis=1)
#
#temp0 = temp[temp[:,0]==0]
#temp0 = temp0[temp0[:,1]==0]
#
#temp1 = temp[temp[:,0]==1]
#temp1 = temp1[temp1[:,1]==0]
#
#temp2 = temp[temp[:,0]==0]
#temp2 = temp2[temp2[:,1]==1]
#
#temp3 = temp[temp[:,0]==1]
#temp3 = temp3[temp3[:,1]==1]
#
#np.mean(temp0[:,2])
#np.mean(temp1[:,2])
#np.mean(temp2[:,2])
#np.mean(temp3[:,2])

###############################################
# Create features for NN
###############################################

def create_nn_features(df, n_hist):
    
    df.sort_values(['Aandeel','Date'])
    
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
