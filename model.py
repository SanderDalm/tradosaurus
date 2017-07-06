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

stocks = pd.read_csv('data/stocks.csv', usecols=['Date', 'Aandeel', 'Close', '1d_beurs', '5d_beurs', '21d_beurs'])
stocks.dropna(inplace=True)

stocks = StockDataFrame.retype(stocks)

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

        temp_df = df[df.aandeel==aandeel]
        _ = temp_df['macd']
        _ = temp_df['rsi_14']

        prices = np.array(temp_df.close.tolist())
        macd = np.array(temp_df.macdh.tolist())
        rsi = np.array(temp_df.rsi_14.tolist())
        ema12 = np.array(temp_df.close_12_ema.tolist())
        ema26 = np.array(temp_df.close_26_ema.tolist())
        ema26 = np.array(temp_df.close_26_ema.tolist())

        beurs1d = np.array(temp_df['1d_beurs'].tolist())
        beurs5d = np.array(temp_df['5d_beurs'].tolist())
        beurs21d = np.array(temp_df['21d_beurs'].tolist())

        cursor = 26
        offset = 10

        while cursor < len(prices)-n_future:

            #if macd[cursor]>macd[cursor-1]>.64:
            #    macd_change=1
            #else:
            #    macd_change=0
            #if ema12[cursor]>ema26[cursor]:
            #    ema_ratio=1
            #else:
            #    ema_ratio=0

            # Create index features from ema
            macd_change = macd[cursor]-macd[cursor-1]
            rsi_feature = rsi[cursor]
            ema_ratio = ema12[cursor]/ema26[cursor]-1
            ema12_diff = ema12[cursor]/ema12[cursor-1]-1
            ema26_diff = ema26[cursor]/ema26[cursor-1]-1

            beurs1d_feature = beurs1d[cursor]
            beurs5d_feature = beurs5d[cursor]
            beurs21d_feature = beurs21d[cursor]

            #            if beurs1d_feature>0:
            #                beurs1d_feature = 1
            #            else:
            #                beurs1d_feature = 0
            #
            #            if beurs5d_feature>0:
            #                beurs5d_feature = 1
            #            else:
            #                beurs5d_feature = 0
            #
            #            if beurs21d_feature>0:
            #                beurs21d_feature = 1
            #            else:
            #                beurs21d_feature = 0

            features.append([macd_change, rsi_feature, ema_ratio, ema12_diff, ema26_diff])#, beurs1d_feature, beurs5d_feature, beurs21d_feature])

            # Calculate price change
            label = prices[cursor+n_future]/prices[cursor]

            labels.append(label)
            cursor += offset

    return np.array(features), np.array(labels)


###############################################
# Fit models
###############################################

features_train, labels_train = create_filter_features(stocks_train, 2)
features_test, labels_test = create_filter_features(stocks_test, 2)

model = RandomForestRegressor(n_estimators=100)
#model = LinearRegression()
model.fit(features_train, labels_train)
model.score(features_test, labels_test)
plt.scatter(model.predict(features_test), labels_test)


temp = np.concatenate([features_train, labels_train.reshape([len(labels_train),1])], axis=1)

# Dit werkt?
temp0 = temp.copy()
temp0 = temp0[temp0[:,1]>10]
temp0 = temp0[temp0[:,1]<30]


np.mean(temp[:,-1])
np.mean(temp0[:,-1])

plt.scatter(temp[:,1], temp[:,-1])


###############################################
# Create features for NN
###############################################

def create_nn_features(df, n_hist):

    df.sort_values(['aandeel','date'])

    features = []

    for aandeel in tqdm(df.aandeel.unique().tolist()):

        temp_df = df[df.aandeel==aandeel]

        prices = np.array(temp_df.close.tolist())

        cursor = 0
        offset = n_hist

        while cursor < len(prices)-n_hist:

            feature = prices[cursor:cursor+n_hist]
            feature = feature-np.mean(feature)
            feature = feature/np.std(feature)
            features.append(feature)

            cursor += offset

    return np.array(features)


features_train = create_nn_features(stocks_train, 100)