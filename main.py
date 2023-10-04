import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import binance

# Set the Binance API endpoint for ETHUSDT
url = 'url_for_APIs_with_latest_version'

# Set your Binance API key
client = binance.Client(api_key='API_Key', api_secret='Secret_Key')

# Make the API request
response = client.get(url)

# Convert the response to JSON
data = response.json()

# Extract the price data
price_data = data

# Convert the price data to a NumPy array
price_array = np.array(price_data)

# Split the data into training and test sets
X_train = price_array[:-100]
y_train = price_array[1:-99]
X_test = price_array[-100:]

# Create the bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])), merge_mode='concat'))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100)

# Make a prediction for the next day
current_price = price_array[-1][-1]
date_time = datetime(2023, 10, 3, 12)
X_new = np.array([[current_price]])
y_pred_new = model.predict(X_new)

# Print the prediction
print('Prediction for next day:', y_pred_new)
