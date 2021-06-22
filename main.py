"""
Short term arbitrage with LSTM (~1 day)

"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers Dense, Dropout, LSTM


# Load Data
# Ticket symbol to load data
company = 'AAPL'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)


data = web.DataReader(company, 'yahoo', start, end)


# Prepare data 
# Fit data into 0 to 1 using sk.learnpreprocessor module
scaler = MinMaxScaler(feature_range(0, 1))
# Predict closing price
scalred_data = scaler.fit_transform(data['Close'].values)