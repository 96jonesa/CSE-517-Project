import torch
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


def prep_dataset(dataset_filepath, start_date, end_date):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

    tf.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

    cache = {}
    calendar = mcal.get_calendar('NYSE')
    def next_trading_day(start_day=None, SAFE_DELTA = 4):
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

    raw_prices_filepath = stocknet_dataset_filepath + '/price/raw'
    preprocessed_tweets_filepath = stocknet_dataset_filepath + '/tweet/preprocessed'

    company_to_price_df = {}
    company_to_tweets = {}

    for filename in os.listdir(raw_prices_filepath):
        with open(raw_prices_filepath + '/' + filename) as file:
            company_name = filename.split('.')[0]

            # Not enough data for GMRE
            if company_name == 'GMRE':
                continue
            df = pd.read_csv(file)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'adjust_close', 'volume']
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df = df.loc[mask]
            company_to_price_df[company_name] = df.dropna()

    for filename in tqdm(os.listdir(preprocessed_tweets_filepath)):
        company_name = filename.split('.')[0]
        dates_to_tweets = {}
        for tweet_filename in os.listdir(preprocessed_tweets_filepath + '/' + filename):
            if tweet_filename < start_date or tweet_filename > end_date:
                continue
            with open(preprocessed_tweets_filepath + '/' + filename + '/' + tweet_filename) as file:
                list_of_tweets = []
                for line in file:
                    tweet_json = json.loads(line)
                    list_of_tweets.append(tweet_json)
                date_idx = next_trading_day(tweet_filename)
                if date_idx not in dates_to_tweets:
                    dates_to_tweets[date_idx] = list_of_tweets
                else:
                    dates_to_tweets[date_idx] += list_of_tweets
        company_to_tweets[company_name] = dates_to_tweets

    # Reduce logging output.
    logging.set_verbosity(logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Import the Universal Sentence Encoder's TF Hub module
    def embed_useT(module):
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})
    embed_fn = embed_useT(module_url)

    # Generate embeddings
    for company in tqdm(company_to_tweets.keys()):
        for date in company_to_tweets[company].keys():
            messages = []
            for j in range(len(company_to_tweets[company][date])):
                messages.append(' '.join(company_to_tweets[company][date][j]['text']))
                message_embeddings = embed_fn(messages)
            for k in range(len(company_to_tweets[company][date])):
                company_to_tweets[company][date][k]['embedding'] = list(message_embeddings[k])

    # Create date mapping
    date_universe = set()
    for company in company_to_price_df.keys():
        date_universe = date_universe.union(set(company_to_price_df[company].date))
    for company in company_to_tweets.keys():
        date_universe = date_universe.union(set(company_to_tweets[company].keys()))
    date_universe = sorted(list(date_universe))
    index_to_date = {i-5:d for i,d in enumerate(date_universe)}
    date_to_index = {d:i-5 for i,d in enumerate(date_universe)}

    # Calculate dimensions for tensor
    n_stocks = len(company_to_tweets.keys())
    n_days = len(date_universe)
    max_tweets = 0
    for c,d in itertools.product(company_to_tweets.keys(), date_universe):
        if d in company_to_tweets[c]:
            max_tweets = max(max_tweets, len(company_to_tweets[c][d]))
    # Create index mapping for stocks alphabetically
    company_to_index = {c:i for i,c in enumerate(sorted(list(company_to_tweets.keys())))}

    return company_to_price_df, company_to_tweets, date_universe, n_days, n_stocks, max_tweets
