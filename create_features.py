import os

from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import matplotlib.pyplot as plt

###############################################
# Read stock data
###############################################

stocks = pd.read_csv('data/stocks.csv',
                     usecols=['Date', 'Aandeel', 'Volume', 'Close', 
                     'exchange_total', '1d_beurs', '5d_beurs', '21d_beurs'])

stocks.dropna(inplace=True)

stocks.sort_values('Date', inplace=True)


stocks = StockDataFrame.retype(stocks)

#stock_list = np.array(stocks.aandeel.unique().tolist())

#np.random.shuffle(stock_list)
#train_stocks = stock_list[:446]
#test_stocks = stock_list[446:]
#
#stocks_train = stocks[stocks.aandeel.isin(train_stocks)]
#stocks_test = stocks[stocks.aandeel.isin(test_stocks)]
#
#stocks_train['date'] = stocks_train.index
#stocks_test['date'] = stocks_test.index

stocks['date'] = stocks.index
stocks_train = stocks[stocks['date'] < '2016-01-01']
stocks_test = stocks[stocks['date'] >= '2016-01-01']
stocks_train['date'] = stocks_train.index
stocks_test['date'] = stocks_test.index


def standardize(array):    
    mean = np.mean(array)
    std = np.std(array)    
    return (array-mean)/std

def indexize(array):    
    array = (array/array[0])
    return array*100-100



###############################################
# Create features for filtering
###############################################

def get_random_forest_features(n_future, n_hist, offset):
    
    assert n_hist>26,'History must be >26 periods'
    
    def trend(array):
        
        downtrend = 1
        for index, value in enumerate(array):
            
            if index == 0:
                continue
            else:
                if array[index]<=array[index-1]:
                    continue
                else:
                    downtrend = 0
            
        lowest_point = 0
        if np.argmin(array) == len(array)-1:        
            lowest_point = 1
        
        return downtrend, lowest_point
            
    
    def create_filter_features(df, n_future, offset, n_hist):
    
        df.sort_values(['aandeel','date'])
    
        features = []
        labels = []
        price_history_list = []
    
        for aandeel in tqdm(df.aandeel.unique().tolist()):
            
            temp_df = df[df.aandeel==aandeel]
            
            _ = temp_df['macd']
            _ = temp_df['rsi_14']
    
            price_list = np.array(temp_df.close.tolist())
            macd_list = np.array(temp_df.macdh.tolist())
            rsi_list = np.array(temp_df.rsi_14.tolist())
            ema12_list = np.array(temp_df.close_12_ema.tolist())
            ema26_list = np.array(temp_df.close_26_ema.tolist())
            ema_ratio_list = ema12_list/ema26_list-1
                             
            volume_list = np.array(temp_df.volume.tolist())
                    
    
            beurs1d_list = np.array(temp_df['1d_beurs'].tolist())
            beurs5d_list = np.array(temp_df['5d_beurs'].tolist())
            beurs21d_list = np.array(temp_df['21d_beurs'].tolist())
    
            cursor = n_hist            
            while cursor < len(price_list)-n_future:
                
                price_history = np.array(price_list[cursor-n_hist:cursor])
                                
                downtrend26, lowest26 = trend(price_list[cursor-26:cursor])
                downtrend7, lowest7 = trend(price_list[cursor-7:cursor])
                
                uptrend26, highest26 = trend(np.flip(price_list[cursor-26:cursor],0))
                uptrend7, highest7 = trend(np.flip(price_list[cursor-7:cursor],0))
                    
                
                # Create index features from ema
                macd = macd_list[cursor]
                macd_diff = macd_list[cursor]-macd_list[cursor-1]
                            
                price_diff = price_list[cursor]/price_list[cursor-1]-1
                volume_diff = volume_list[cursor]/volume_list[cursor-1]-1
                                    
                rsi = rsi_list[cursor]/100
                ema_ratio = ema_ratio_list[cursor]
                
                ema12_diff = ema12_list[cursor]/ema12_list[cursor-1]-1
                ema26_diff = ema26_list[cursor]/ema26_list[cursor-1]-1
                        
    
                beurs1d = beurs1d_list[cursor]
                beurs5d = beurs5d_list[cursor]
                beurs21d = beurs21d_list[cursor]
    
    
                features.append([macd, macd_diff, price_diff, volume_diff,
                                 rsi, ema_ratio, ema12_diff, ema26_diff,
                                 downtrend26, lowest26, downtrend7, lowest7,
                                 uptrend26, highest26, uptrend7, highest7,
                                 beurs1d, beurs5d, beurs21d])
    
                # Calculate price change
                label = price_list[cursor+n_future]/price_list[cursor]-1
    
                labels.append(label)
                price_history_list.append(price_history)
                cursor += offset
    
        return np.array(features), np.array(labels), np.array(price_history_list)
        
    features_train, labels_train, price_history_train = create_filter_features(stocks_train, n_future, offset, n_hist)
    features_test, labels_test, price_history_test = create_filter_features(stocks_test, n_future, offset, n_hist)
    
    features_train = np.array([standardize(x) for x in features_train[:,]])
    labels_train = standardize(labels_train)
    #price_history_train = np.array([standardize(x) for x in price_history_train[:,]])
    
    features_test = np.array([standardize(x) for x in features_test[:,]])
    labels_test = standardize(labels_test)
    #price_history_test = np.array([standardize(x) for x in price_history_test[:,]])
    
    features_train = features_train[np.where(abs(labels_train)<3)[0]]
    price_history_train = price_history_train[np.where(abs(labels_train)<3)[0]]
    labels_train = labels_train[np.where(abs(labels_train)<3)[0]]
    
    features_test = features_test[np.where(abs(labels_test)<3)[0]]
    price_history_test = price_history_test[np.where(abs(labels_test)<3)[0]]
    labels_test = labels_test[np.where(abs(labels_test)<3)[0]]
    
    features_train = np.concatenate([price_history_train, features_train], axis=1)
    features_test = np.concatenate([price_history_test, features_test], axis=1)
    
    return features_train, labels_train, price_history_train, features_test, labels_test, price_history_test

###############################################
# Create features for NN
###############################################

def get_nn_features(n_hist, n_future, offset, train_dir, test_dir, price_pred):
    
    # Delete old data    
    train_data_list = glob(train_dir+'*.npy')        
    test_data_list = glob(test_dir+'*.npy')        
            
    for item in train_data_list:
       os.remove(item)
    for item in test_data_list:
        os.remove(item)
        
    def create_nn_features(df, n_hist, n_future, offset, outdir, price_pred):
    
        df.sort_values(['aandeel', 'date'])
    
        for aandeel in tqdm(df.aandeel.unique().tolist()):
    
            temp_df = df[df.aandeel==aandeel]
    
            price_list = np.array(temp_df.close)
            volume_list = np.array(temp_df.volume)
            exchange_list = np.array(temp_df.exchange_total)
    
            cursor = 0        
    
            while cursor < len(price_list)-n_hist:
    
                price_history = price_list[cursor:cursor+n_hist]
                np.save(outdir+aandeel+str(cursor)+'_price_history', price_history)
                
                price_history = indexize(price_history)
                volume_history = indexize(volume_list[cursor:cursor+n_hist])
                exchange_history = indexize(exchange_list[cursor:cursor+n_hist])
                
                x = np.concatenate([price_history, volume_history, exchange_history], axis=0).reshape([3, n_hist])
                            
                # Direct price prediction
                if price_pred:
                    y = x[0,n_future:].reshape([1, n_hist-n_future])
                    x = x[:,:-n_future]
                    
                else:
                    x_next = x[:,n_future:]                                
                    x_diff = x_next-x[:,:-n_future]
                    y = x_diff[:,n_future:]
                    x_diff = x_diff[:,:-n_future]                                                
                    y = y[0,:].reshape([1, n_hist-n_future*2]) 
            
                                
                #noise = np.random.normal(0,.5, [x_diff.shape[0], x_diff.shape[1]])                                
                #x_diff = noise
                
                # Price only
                #x_diff = x_diff[0].reshape([1,n_future-n_hist*2])                
                
                if price_pred:                    
                    features = np.concatenate([x, y], axis=0)        
                else:
                    features = np.concatenate([x_diff, y], axis=0)        
                np.save(outdir+aandeel+str(cursor), features)
                
                cursor += offset
        

    create_nn_features(stocks_train, n_hist, n_future, offset, train_dir, price_pred)    
    create_nn_features(stocks_test, n_hist, n_future, offset, test_dir, price_pred)
