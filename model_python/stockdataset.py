import torch
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


class StockDataset(Dataset):
    """Price dataset"""

    def __init__(self, company_to_price_df, company_to_tweets, date_universe, n_days, n_stocks, max_tweets):
        # Initialize class members
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.max_tweets = max_tweets
        self.window = 6
        window = self.window

        # Build maps
        self.company_to_index = {c:i for i,c in enumerate(sorted(list(company_to_tweets.keys())))}
        self.date_to_index = {d:i for i,d in enumerate(date_universe)}
        self.index_to_date = {i:d for i,d in enumerate(date_universe)}

        # Store data
        self.company_to_price_df = company_to_price_df
        self.company_to_tweets = company_to_tweets
        
        # Get price data tensor: n_stocks, n_days, 3
        self.price_data = np.zeros((n_stocks, n_days, 3))
        for company in company_to_price_df.keys():
            df = company_to_price_df[company]
            df.reset_index(inplace=True, drop=True)
            # Look up specific rows in DF
            for index, row in df.iterrows():
                # Grab row with particular date
                if index != 0:
                    d_index = self.date_to_index[row['date']]
                    c_index = self.company_to_index[company]
                    self.price_data[c_index, d_index, 0] = row['high'] / prev_close
                    self.price_data[c_index, d_index, 1] = row['low'] / prev_close
                    self.price_data[c_index, d_index, 2] = row['close'] / prev_close
                prev_close = row['close']
                
        # Which stocks are usable for these dates, shape n_days n_stocks
        self.usable_stocks = torch.ones((self.n_days-7, self.n_stocks))
    
        # Labels of shape n_days, n_stocks
        self.labels = torch.zeros((self.n_days-7, self.n_stocks))
        

        # Get labels
        for i in range(self.n_days-7):
            # Day after (for label)
            day_after = self.index_to_date[i + window + 1]
            # Current day
            current_day = self.index_to_date[i + window]
            for company in self.company_to_price_df.keys():
                df = self.company_to_price_df[company]

                # Grab row with particular date
                post_row = df.loc[df['date'] == day_after]
                row = df.loc[df['date'] == current_day]
                c_index = self.company_to_index[company]

                if (len(post_row['close']) > 0) and (len(row['close']) > 0):
                    close = np.zeros((1))
                    close[0] = post_row['close']
                    close[0] /= row['close']
                    if close >= 1.0055:
                        self.labels[i, c_index] = 1
                    elif close <= 0.995:
                        self.labels[i, c_index] = 0
                    else:
                        self.usable_stocks[i, c_index] = 0
                else:
                    self.usable_stocks[i, c_index] = 0

    def __len__(self):
        return self.n_days-7

    def __getitem__(self, idx):
        """
        gets a price tensor of shape (n_stocks, 6, 3)
        gets a smi tensor of shape (n_stocks, 6, K, 512)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Size of sliding window
        window = self.window
        
        # Current day's usable stocks from price filter
        usable_stocks = self.usable_stocks[idx]
        
        # Labels from price day
        labels = self.labels[idx]

        # Dates that we need to look up
        dates_range = [self.index_to_date[i] for i in range(idx + 1, idx + window + 1)]

        # Day after (for label)
        day_after = self.index_to_date[idx + window + 1]

        # Current day
        current_day = self.index_to_date[idx + window]

        # Get price data tensor: n_stocks, window, 3
        price_data = self.price_data[:, idx+1:idx+window+1, :]

        # Extract tweets for specific window
        smi_data = np.zeros((self.n_stocks, window, self.max_tweets, 512))
        tweet_counts = np.zeros((self.n_stocks, window))
        for company in self.company_to_tweets.keys():

            # Look up tweets from specific days
            for date_idx, date in enumerate(dates_range):
                n_tweets = 0
                tweets = []
                c_index = self.company_to_index[company]
                if date in self.company_to_tweets[company]:
                    n_tweets = len(self.company_to_tweets[company][date])
                    tweets = [self.company_to_tweets[company][date][k]['embedding'] for k in range(n_tweets)]
                else:
                    usable_stocks[c_index] = 0
                tweet_counts[c_index, date_idx] = n_tweets
                if n_tweets == 0:
                    usable_stocks[c_index] = 0
                for i,embedding in enumerate(tweets): 
                    #stocks, day, lags, tweet, embedding
                    smi_data[c_index, date_idx, i, :] = embedding[:]

        usable_stocks = (usable_stocks == 1)

        m_mask = torch.zeros(6, self.n_stocks, self.max_tweets, 1)
        for t in range(6):
            for i in range(self.n_stocks):
                m_mask[t, i, 0:int(round(tweet_counts[i][t])), 0] = 1

        price_output = price_data[usable_stocks,:,:]
        smi_output = smi_data[usable_stocks,:,:,:]
        tweet_count = tweet_counts[usable_stocks,:]
        m_mask = m_mask[:,usable_stocks,:,:]
        labels = labels[usable_stocks]
        
        # construct output
        return price_output, smi_output, tweet_count, usable_stocks, labels, m_mask
