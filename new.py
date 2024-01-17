# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('ETH-USD.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Ethereum Price')
plt.title('Ethereum Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ARIMA model
model_arima = ARIMA(train, order=(5,1,0))
fit_arima = model_arima.fit()

# Make predictions on the test set
predictions_arima = fit_arima.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Evaluate ARIMA model
mse_arima = mean_squared_error(test, predictions_arima)
mape_arima = calculate_mape(test, predictions_arima)
print(f'ARIMA Model - Mean Squared Error: {mse_arima}, MAPE: {mape_arima}%')

# LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Reshape data for LSTM
X_train = np.reshape(train_scaled, (train_scaled.shape[0], 1, 1))
X_test = np.reshape(test_scaled, (test_scaled.shape[0], 1, 1))

# Create and train the LSTM model
model_lstm = create_lstm_model()
model_lstm.fit(X_train, train_scaled, epochs=50, batch_size=1, verbose=2)

# Make predictions on the test set
predictions_lstm_scaled = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm_scaled)

# Evaluate LSTM model
mse_lstm = mean_squared_error(test, predictions_lstm)
mape_lstm = calculate_mape(test, predictions_lstm)
print(f'LSTM Model - Mean Squared Error: {mse_lstm}, MAPE: {mape_lstm}%')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Prices')
plt.plot(test.index, predictions_arima, label='ARIMA Predictions')
plt.plot(test.index, predictions_lstm, label='LSTM Predictions')
plt.title('Ethereum Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
