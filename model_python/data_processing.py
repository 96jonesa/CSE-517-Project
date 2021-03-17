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

    raw_prices_filepath = dataset_filepath + '/price/raw'
    preprocessed_tweets_filepath = dataset_filepath + '/tweet/preprocessed'

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

def build_second_order_wikidata_graphs(wikidata_filepath):
    wikidata_entries_filepath = os.path.join(wikidata_filepath, 'wikidata_entries')
    # Entity numbers
    # NOTE: SPLP could not be found on Wikidata
    name_to_num = {}
    num_to_name = {}
    entities = set()
    with open(os.path.join(wikidata_filepath, 'links.csv')) as file:
        next(file)
        for line in file:
            name, url = line.split(',')
            entity_num = int(re.sub('\D', '', url.split(',')[-1]))
            name_to_num[name] = entity_num
            num_to_name[entity_num] = name
            entities.add(entity_num)
        # Building first-order relationships
        graph = {}
        for name in name_to_num.keys():
            graph[name] = []
        graph['SPLP'] = []

    for filename in os.listdir(wikidata_entries_filepath):
        with open(os.path.join(wikidata_entries_filepath, filename)) as file:
            if filename == '.DS_Store': continue
                
            orig_entity = filename.split('_')[0]
            # print('\nSearching entity', orig_entity)
            
            for e_id in entities:
                file.seek(0, 0)
                if str(e_id) in file.read():
                    # print(f'{orig_entity} contains entity {num_to_name[e_id]}')
                    graph[orig_entity].append(num_to_name[e_id])

    # Make graph non-directional
    for co1 in graph.keys():
        co2s = graph[co1]
        for co2 in co2s:
            if co1 not in graph[co2]:
                graph[co2].append(co1)

    # Build graph mapping companies to all their related entities
    entity_regex = re.compile(".+Q[0-9].+")

    company_to_entities = {}
    for name in name_to_num.keys():
        company_to_entities[name] = []
    company_to_entities['SPLP'] = []

    for filename in os.listdir(wikidata_entries_filepath):
        with open(os.path.join(wikidata_entries_filepath, filename)) as file:
            if filename == '.DS_Store': continue
                
            orig_entity = filename.split('_')[0]
            # print('\nSearching entity', orig_entity, name_to_num[orig_entity])
            
            for line in file:
                if entity_regex.match(line):
                    try:
                        q_index = line.index('Q')
                        s_index = line[q_index:].index(' ') + q_index
                        related_entity = line[q_index + 1:s_index]
                        
                        if '-' not in related_entity:
                            # print('>', related_entity)
                            if related_entity not in company_to_entities[orig_entity]:
                                company_to_entities[orig_entity].append(related_entity)
                    except ValueError:
                        # print('substring err')
                        pass

    # Build second-order relations
    graph_2 = {}
    for name in name_to_num.keys():
        graph_2[name] = set()
    graph_2['SPLP'] = set()

    def common_member(a, b): 
        a_set = set(a) 
        b_set = set(b) 
        if len(a_set.intersection(b_set)) > 0: 
            return(True)  
        return(False)  

    for company in company_to_entities.keys():
        for other_company in company_to_entities.keys():
            if common_member(company_to_entities[company], company_to_entities[other_company]):
                graph_2[company].add(other_company)

    # Make graph non-directional
    for co1 in graph_2.keys():
        co2s = graph_2[co1]
        for co2 in co2s:
            if co1 not in graph_2[co2]:
                graph[co2].add(co1)

    # First and second order graphs combined
    graph_1_2 = {}
    for name in name_to_num.keys():
        graph_1_2[name] = set()
    graph_1_2['SPLP'] = set()

    for company in graph_1_2.keys():
        graph_1_2[company].update(graph[company])
        graph_1_2[company].update(graph_2[company])

    return graph_1_2