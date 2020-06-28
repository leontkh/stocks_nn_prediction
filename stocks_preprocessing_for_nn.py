from collections import Counter
import datetime as dt
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import torch

#Retrieves stock data from Yahoo Finance
def get_data_from_yahoo(ticker, duration, refresh=False):
    
    if not os.path.exists('stocks_nn'):
        os.makedirs('stocks_nn')

    start = dt.datetime.today() - dt.timedelta(days = duration+1)
    end = dt.datetime.today() - dt.timedelta(days=1)
    print(start)
    if not os.path.exists('stocks_nn/{}.csv'.format(ticker)) or refresh:
        try:
            df = web.DataReader("{}".format(ticker), 'yahoo', start, end)
            df.to_csv('stocks_nn/{}.csv'.format(ticker))
            print('Data retrieved from Yahoo! Finance!')
        except:
            print('Failed to get {} data'.format(ticker))
    else:
        print('Already have {}'.format(ticker))

#Condition for preparing labels
def buy_sell_hold(*args):
    cols = [c for c in args]
    upRequirement = 0.02435
    downRequirement = 0.022405
    for col in cols:
        if col > upRequirement:
            return 2
        if col < -downRequirement:
            return 0
    return 1

#Preparing label set
def process_data_for_labels(ticker, duration=365*3, refresh=False):
    if not os.path.exists('stocks_nn/{}.csv'.format(ticker)) or refresh:
        get_data_from_yahoo(ticker, duration, refresh=True)
    try:
        df = pd.read_csv('stocks_nn/{}.csv'.format(ticker), index_col=0)
    except:
        print("An error has been encountered")
    hm_days = 14
    df = pd.read_csv('stocks_nn/{}.csv'.format(ticker), index_col=0)
    df.dropna(inplace=True)
    df = df[(df != 0).all(1)]

    for i in range(1, hm_days+1):
        df['Adj Close_{}d'.format(i)] = (df['Adj Close'].shift(-i) - df['Adj Close']) /df['Adj Close'].shift(-i)

    df['Adj Close_target'.format(ticker)] = list(map( buy_sell_hold,
                                                df['Adj Close_{}d'.format(1)],
                                                df['Adj Close_{}d'.format(2)],
                                                df['Adj Close_{}d'.format(3)],
                                                df['Adj Close_{}d'.format(4)],
                                                df['Adj Close_{}d'.format(5)],
                                                df['Adj Close_{}d'.format(6)],
                                                df['Adj Close_{}d'.format(7)],
                                                df['Adj Close_{}d'.format(8)],
                                                df['Adj Close_{}d'.format(9)],
                                                df['Adj Close_{}d'.format(10)],
                                                df['Adj Close_{}d'.format(11)],
                                                df['Adj Close_{}d'.format(12)],
                                                df['Adj Close_{}d'.format(13)],
                                                df['Adj Close_{}d'.format(14)],
                                                ))
    labels = torch.tensor(df['Adj Close_target'.format(ticker)].values[14:-14])
    print('Data spread:', Counter(df['Adj Close_target'.format(ticker)].values[14:-14]))
    torch.save(labels, 'stocks_nn/stock_labels.pt')

#Preparing data set
def process_data_for_data(ticker, duration=365*3, refresh=False):
    if not os.path.exists('stocks_nn/{}.csv'.format(ticker)) or refresh:
        get_data_from_yahoo(ticker, duration, refresh=True)
    try:
        df = pd.read_csv('stocks_nn/{}.csv'.format(ticker), index_col=0)
    except:
        print("An error has been encountered")
        
    df = pd.read_csv('stocks_nn/{}.csv'.format(ticker), index_col=0)
    df.dropna(inplace=True)
    data = torch.empty(0)
    for i in range(len(df['Adj Close'][:-14])): # consist of [14:-14] data points for train and test+ another 14 points to predict
        batch = torch.tensor(df[['Adj Close','Open','High','Low','Close']].values[i:i+14]).float()
        row = batch.unsqueeze(0).unsqueeze(0)
        data = torch.cat((data, row),0)
    df.dropna(inplace=True)
    df = df[(df != 0).all(1)]
    torch.save(data, 'stocks_nn/stock_data.pt')
    

if __name__ == "__main__":
    process_data_for_labels("^GSPC", duration=10000, refresh=True)
    process_data_for_data("^GSPC")
