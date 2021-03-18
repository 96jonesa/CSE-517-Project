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


evaluate_model(mansf, test_dataloader):
    # set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # move model to device
    mansf = mansf.to(device)

    mansf.eval()
    correct = 0.0
    total = 0.0
    for price, smi, n_tweets, usable_stocks, labels, m_mask in tqdm(test_dataloader):
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

    return test_acc
