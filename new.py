import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
csv_file = "Ethereum History(2019-2023).csv"
data = pd.read_csv(csv_file)
data['Date'] = pd.to_datetime(data['Date'])

# Used 'Close' prices for prediction
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(closing_prices)

# Convert time series to supervised learning problem
X, y = [], []
for i in range(60, len(normalized_data) - 1):  # using last 60 days to predict next day
    X.append(normalized_data[i-60:i, 0])
    y.append(normalized_data[i, 0])
X, y = np.array(X), np.array(y)

