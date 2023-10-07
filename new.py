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

