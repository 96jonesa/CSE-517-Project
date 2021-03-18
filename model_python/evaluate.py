import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import json
import itertools
import pandas as pd
import torch
import pandas_market_calendars as mcal
import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import MANSF
from stockdataset import StockDataset
from data_processing import prep_dataset

path = '../stocknet-dataset-master/price/raw'
def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    
    """Iterate over each symbol"""
    for symbol in symbols:
        """Import new data"""
        df_temp = pd.read_csv(os.path.join(path, symbol+'.csv'), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
    return df

cache = {}
calendar = mcal.get_calendar('NYSE')
def next_trading_day(start_day=None, SAFE_DELTA = 5):
    """Returns the next/previous trading date separated by a certain number of 
    trading days.
    """
    if start_day is None:
        start_day = datetime.datetime.utcnow().date()
    if start_day in cache:
        return cache[start_day]
    start = pd.to_datetime(start_day)
    end = start + np.timedelta64(SAFE_DELTA, 'D')
    business_days = calendar.valid_days(start_date=start, end_date=end)
    next_day = business_days[1].date()
    next_day = next_day.strftime("%Y-%m-%d")
    cache[start_day] = next_day
    return next_day

# Calculate value change of portfolio assuming the order file and equal cash allocation
def compute_portvals(df_orders, start_val = 100, commission=0, impact=0):
    # prep variables and date range
    df_orders.sort_index(inplace=True)
    start_date = df_orders.index[0]
    end_date = next_trading_day(df_orders.index[-1])
    symbols = list(set(df_orders.columns.values))
    
    # get price data
    df_prices = get_data(symbols, pd.date_range(start_date, end_date, freq='D')).dropna()
    df_prices = df_prices[symbols]
    df_price_change = df_prices.pct_change()
    df_price_change.iloc[0] = 0
    
    # calculate cumulative holdings
    df_holdings = pd.DataFrame(index=df_prices.index, columns=df_prices.columns.values)
    df_holdings.iloc[0] = 0
    df_holdings.update(df_orders)
    df_holdings = df_holdings.ffill()
    df_port_change = pd.DataFrame(columns=['portfolio'], index=df_price_change.index)
    for index, row in df_holdings.iterrows():
        vals = []
        for col in df_holdings.columns.values:
            if row[col] == 1:
                vals.append(df_price_change.loc[index, col])
            
        if len(vals) == 0:
            vals.append(0)
        df_port_change.loc[index] = sum(vals)/len(vals) + 1
    df_values = df_port_change.cumprod() * start_val
    
    # equal dollar allocation portfolio
    return df_values

def sharpe(portvals, risk_free_rate=0):
    daily_returns = portvals.pct_change().dropna()
    avg_return = np.mean(daily_returns)
    risk = np.std(daily_returns)
    return (np.sqrt(252) * avg_return - risk_free_rate) / risk


def evaluate_model(mansf, test_dataloader, T, company_to_tweets, date_universe):
    # set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get indexes
    index_to_company = {i:c for i,c in enumerate(sorted(list(company_to_tweets.keys())))}
    index_to_date = index_to_date = {i-5:d for i,d in enumerate(date_universe)}
    
    df_orders = pd.DataFrame(columns=sorted(list(index_to_company.values())))

    # move model to device
    mansf = mansf.to(device)

    mansf.eval()
    correct = 0.0
    total = 0.0
    for index, (price, smi, n_tweets, usable_stocks, labels, m_mask) in enumerate(tqdm(test_dataloader)):
        price = price.type(torch.FloatTensor)
        smi = smi.type(torch.FloatTensor)

        price = price.to(device)
        smi = smi.to(device)
        n_tweets = n_tweets.to(device)
        usable_stocks = usable_stocks.to(device)
        labels = labels.to(device)
        m_mask = m_mask.to(device)
        
        stocks = []
        for i, val in enumerate(usable_stocks):
            if val == 1:
                stocks.append(index_to_company[i])

        price = price.view(price.shape[1], price.shape[2], price.shape[3])
        smi = smi.view(smi.shape[1], smi.shape[2], smi.shape[3], smi.shape[4])
        n_tweets = n_tweets.view(n_tweets.shape[1], n_tweets.shape[2])
        usable_stocks = usable_stocks.view(usable_stocks.shape[1])
        m_mask = m_mask.view(m_mask.shape[1], m_mask.shape[2], m_mask.shape[3], m_mask.shape[4])

        smi = smi.permute(1, 0, 2, 3)

        m = []
        for t in range(T):
            m.append(smi[t])

        neighborhoods = torch.eye(87, 87)
        neighborhoods = neighborhoods.to(device)
        neighborhoods = neighborhoods[usable_stocks, :]
        neighborhoods = neighborhoods[:, usable_stocks]

        if price.shape[0] != 0:
            y = mansf(price, smi, m_mask, neighborhoods, device)
            orders = {}
            for i, stock in enumerate(stocks):
                if y[i] == 1:
                    orders[stock] = 1
                elif y[i] == 0:
                    orders[stock] = -1
            df_orders[index_to_date[idx]] = orders
            correct += torch.sum((y > 0.5).view(-1) == labels.view(-1)).item()
            total += len(y)
            
    df_orders = df_orders.fillna(0)
    
    portvals = compute_portvals(df_orders)
    
    sharpe_metric = sharpe(portvals)

    return correct / total, sharpe_metric
