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
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

# Prepare training data
x_train = []
y_train = []


for x  in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    x_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train, np.array(y_train))
x_train = np.reshape(x_train, x_train.shape[0], x_train.shape[1])

#Building model
model = Sequential()


model.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1], 1) ) )












