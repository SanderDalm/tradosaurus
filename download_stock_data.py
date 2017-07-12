import os
import datetime

from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################
# Download stock data
#####################################

def download_and_save_stock_data():
    
    # AEX tickers
    aex = open('tickerlists/aex.txt').read().split('\n')
    aex = [x.strip('\r')+'.AS' for x in aex]
    
    # S&P 500 tickers
    tickers_sp500 = open('tickerlists/sp500.txt').read().split('\n')
    tickers_sp500 = [x.split('\t')[0] for x in tickers_sp500]
    
    # Nasdaq 100 tickers
    tickers_ndx = open('tickerlists/nasdaq100.txt').read().split('\n')
    tickers_ndx = [x.split('\t')[0] for x in tickers_ndx]
    
    tickers = aex[:]
    tickers.extend(tickers_sp500)
    tickers.extend(tickers_ndx)
    
    currentDT = datetime.datetime.now()
    startdate = '2010-01-01'
    today = str(currentDT)[:10]
    
    for ticker in tickers:
        if len(ticker)>0:
            os.system("python get_data.py --symbol={} --from={} --to={} -o {}.csv".format(ticker, startdate, today, ticker))
    
    os.system("python get_data.py --symbol={} --from={} --to={} -o {}.csv".format('^AEX', startdate, today, 'AEX'))
    os.system("python get_data.py --symbol={} --from={} --to={} -o {}.csv".format('^NDX', startdate, today, 'NDX'))
    os.system("python get_data.py --symbol={} --from={} --to={} -o {}.csv".format('^SP500TR', startdate, today, 'SP500TR'))
    
    
    def read_exchange(csv_file):
    
        exchange = pd.read_csv(csv_file, usecols=['Date', 'Close'] , na_values=['null', 0])
        #exchange['Exchange'] = csv_file.strip('data/').strip('.csv')
        exchange.dropna(inplace=True)
        exchange['exchange_total'] = exchange.sum(axis=1)
        exchange['last_price'] = exchange.shift(1)['exchange_total']
        exchange.sort_values('Date', inplace=True)
    
        exchange['1d_hist'] =  exchange.shift(1)['exchange_total']
        exchange['1d_beurs'] =  exchange['exchange_total']/exchange['1d_hist']-1
        exchange['5d_hist'] =  exchange.shift(5)['exchange_total']
        exchange['5d_beurs'] =  exchange['exchange_total']/exchange['5d_hist']-1
        exchange['21d_hist'] =  exchange.shift(21)['exchange_total']
        exchange['21d_beurs'] =  exchange['exchange_total']/exchange['21d_hist']-1
        exchange['250d_hist'] =  exchange.shift(250)['exchange_total']
        exchange['250d_beurs'] =  exchange['exchange_total']/exchange['250d_hist']-1
        exchange = exchange[['Date', 'exchange_total', '1d_beurs', '5d_beurs', '21d_beurs', '250d_beurs']]
    
        return exchange
    
    aex_df = read_exchange('data/AEX.csv')
    ndx_df = read_exchange('data/NDX.csv')
    sp500_df = read_exchange('data/SP500TR.csv')
    
    #####################################
    # Read stock data to df
    #####################################
    
    csv_files = glob('data/'+'*.csv')
    
    stocks = pd.DataFrame(columns=['Aandeel', 'Date', 'Close'])
    
    for csv_file in csv_files:
    
        aandeel = csv_file.strip('data/').strip('.csv')
        if aandeel in tickers:
            print aandeel
            df = pd.read_csv(csv_file, usecols=['Date', 'Volume', 'Close'], na_values=['null', 0])
            df['Aandeel'] = aandeel
            df.dropna(inplace=True)
            df = df[['Aandeel', 'Date', 'Close', 'Volume']]
            if aandeel in aex:
                df['Exchange'] = 'AEX'
            if aandeel in tickers_ndx:
                df['Exchange'] = 'NDX'
            if aandeel in tickers_sp500:
                df['Exchange'] = 'SP500TR'
            stocks = pd.concat([stocks, df])
    
    stocks_aex = pd.merge(stocks[stocks.Exchange=='AEX'], aex_df, how='outer', on=['Date'])
    stocks_ndx = pd.merge(stocks[stocks.Exchange=='NDX'], ndx_df, how='outer', on=['Date'])
    stocks_sp500 = pd.merge(stocks[stocks.Exchange=='SP500TR'], sp500_df, how='outer', on=['Date'])
    
    stocks = pd.concat([stocks_aex, stocks_ndx, stocks_sp500], axis=0)
    
    stocks.sort_values(['Aandeel', 'Date'], inplace=True)
    stocks.to_csv('data/stocks.csv', index=False)
