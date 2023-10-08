import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from the CSV file
csv_file = "ETH-USD.csv"
data = pd.read_csv(csv_file)

# Preprocess the data: Set "Date" as the index and ensure it's in datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use only the "Open" price for prediction
df = data[['Open']].copy()

# Create a new column for the shifted (future) prices
future_days = 5  # Adjust as needed
df['Open_Future'] = df['Open'].shift(-future_days)

# Drop rows with NaN values
df.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Split data into training and test sets
split_ratio = 0.8
split_index = int(len(df_scaled) * split_ratio)
train_data = df_scaled[:split_index]
test_data = df_scaled[split_index:]

# Prepare the training data
X_train, y_train = [], []
for i in range(future_days, len(train_data)):
    X_train.append(train_data[i - future_days:i, 0])
    y_train.append(train_data[i, 1])
X_train, y_train = np.array(X_train), np.array(y_train)

# Prepare the test data
X_test, y_test = [], []
for i in range(future_days, len(test_data)):
    X_test.append(test_data[i - future_days:i, 0])
    y_test.append(test_data[i, 1])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build an LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the test data
predicted_open_prices = model.predict(X_test)

# Inverse transform the scaled predictions
predicted_open_prices = scaler.inverse_transform(np.concatenate((X_test[:, -1:], predicted_open_prices), axis=1))[:, 1]

# Display the predicted open prices
print("Predicted Open Prices:")
print(predicted_open_prices)

# Calculate and display the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predicted_open_prices)
print("Mean Squared Error:", mse)
