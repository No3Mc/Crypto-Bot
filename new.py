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

# Split data into training and validation sets
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Reshaping data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Building LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Function to predict the next day price
def predict_next_day_price(model, last_n_days_data):
    last_n_days_normalized = scaler.transform(last_n_days_data.reshape(-1, 1))
    last_n_days_normalized = np.reshape(last_n_days_normalized, (1, last_n_days_normalized.shape[0], 1))
    predicted_price = model.predict(last_n_days_normalized)
    return scaler.inverse_transform(predicted_price)

