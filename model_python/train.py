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


# specify hyperparameter values
T = 6
GRU_HIDDEN_SIZE = 64
ATTN_INTER_SIZE = 32
USE_EMBED_SIZE = 512
BLEND_SIZE = 32
GAT_1_INTER_SIZE = 32
GAT_2_INTER_SIZE = 32
LEAKYRELU_SLOPE = 0.01
ELU_ALPHA = 1.0
U = 8
LEARNING_RATE = 5e-4
NUM_EPOCHS = 25

# specify data split
stocknet_dataset_filepath = './stocknet-dataset-master'
train_start_date = '2014-01-01'
train_end_date = '2015-07-31'
val_start_date = '2015-08-01'
val_end_date = '2015-09-30'

# prepare data
train_company_to_price_df, train_company_to_tweets, train_date_universe, train_n_days, train_n_stocks, train_max_tweets = prep_dataset(stocknet_dataset_filepath, train_start_date, train_end_date)
val_company_to_price_df, val_company_to_tweets, val_date_universe, val_n_days, val_n_stocks, val_max_tweets = prep_dataset(stocknet_dataset_filepath, val_start_date, val_end_date)
test_company_to_price_df, test_company_to_tweets, test_date_universe, test_n_days, test_n_stocks, test_max_tweets = prep_dataset(stocknet_dataset_filepath, test_start_date, test_end_date)

# create datasets
train_dataset = StockDataset(train_company_to_price_df, train_company_to_tweets, train_date_universe, train_n_days, train_n_stocks, train_max_tweets)
val_dataset = StockDataset(val_company_to_price_df, val_company_to_tweets, val_date_universe, val_n_days, val_n_stocks, val_max_tweets)
test_dataset = StockDataset(test_company_to_price_df, test_company_to_tweets, test_date_universe, test_n_days, test_n_stocks, test_max_tweets)

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=1,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(val_dataset, batch_size=1,
                        shuffle=False, num_workers=0)

test_dataloader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0)

# set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create model
mansf = MANSF(T=T,
              gru_hidden_size=GRU_HIDDEN_SIZE,
              attn_inter_size=ATTN_INTER_SIZE,
              use_embed_size=USE_EMBED_SIZE,
              blend_size=BLEND_SIZE,
              gat_1_inter_size=GAT_1_INTER_SIZE,
              gat_2_inter_size=GAT_2_INTER_SIZE,
              leakyrelu_slope=LEAKYRELU_SLOPE,
              elu_alpha=ELU_ALPHA,
              U=U)

# move model to device
mansf = mansf.to(device)

# create optiminzer and loss function
optimizer = optim.Adam(mansf.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction='sum')

# lists for tracking accuracy across training epochs
train_acc_list = []
val_acc_list = []

# train model
for epoch in range(25):
    mansf.train()
    correct = 0.0
    total = 0.0
    running_loss = 0.0
    for price, smi, n_tweets, usable_stocks, labels, m_mask in tqdm(train_dataloader):
        price = price.type(torch.FloatTensor)
        smi = smi.type(torch.FloatTensor)

        price = price.to(device)
        smi = smi.to(device)
        n_tweets = n_tweets.to(device)
        usable_stocks = usable_stocks.to(device)
        labels = labels.to(device)
        m_mask = m_mask.to(device)

        price = price.squeeze()
        smi = smi.squeeze()
        n_tweets = n_tweets.squeeze()
        usable_stocks = usable_stocks.squeeze()
        m_mask = m_mask.squeeze()

        smi = smi.permute(1, 0, 2, 3)

        m = []
        for t in range(T):
            m.append(smi[t])

        neighborhoods = torch.eye(87, 87)
        neighborhoods = neighborhoods.to(device)
        neighborhoods = neighborhoods[usable_stocks, usable_stocks]

        if price.shape[0] != 0:
            y = mansf(price, smi, m_mask, neighborhoods)
            loss = loss_fn(y.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            correct += torch.sum(((y > 0.5).view(-1) == labels.view(-1))).item()
            total += len(y)
            running_loss = loss.item() * len(y)

    train_acc = correct / total
    train_acc_list.append(train_acc)

    mansf.eval()
    correct = 0.0
    total = 0.0
    for price, smi, n_tweets, usable_stocks, labels, m_mask in tqdm(val_dataloader):
        price = price.type(torch.FloatTensor)
        smi = smi.type(torch.FloatTensor)

        price = price.to(device)
        smi = smi.to(device)
        n_tweets = n_tweets.to(device)
        usable_stocks = usable_stocks.to(device)
        labels = labels.to(device)
        m_mask = m_mask.to(device)

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
        neighborhoods = neighborhoods[usable_stocks, usable_stocks]

        if price.shape[0] != 0:
            y = mansf(price, smi, m_mask, neighborhoods)
            correct += torch.sum((y > 0.5).view(-1) == labels.view(-1)).item()
            total += len(y)

    val_acc = correct / total
    val_acc_list.append(val_acc)

    print('epoch:', epoch, 'loss:', running_loss, 'train_acc:', train_acc, 'val_acc:', val_acc)
