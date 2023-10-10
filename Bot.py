import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from the CSV file
csv_file = "ETH-USD.csv"  # Replace with the path to your Ethereum data CSV file
data = pd.read_csv(csv_file)

# Preprocess the data: Set "Date" as the index and ensure it's in datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define features (X) and target variable (y)
X = data[["High", "Low", "Close", "Adj Close", "Volume"]]  # Features
y = data["Open"]  # Target variable

# Initialize and train the linear regression model on the entire dataset
regr = LinearRegression()
regr.fit(X, y)

# Define a range of future dates for prediction
future_dates = pd.date_range(start="2023-10-07", end="2023-10-11")  # Example date range

# Make predictions for each future date, adjusting the price based on the previous prediction
predicted_open_prices = []

previous_prediction = None  # Initialize previous prediction to None

for input_date in future_dates:
    try:
        # Create a feature vector for prediction
        input_features = X.loc[input_date].values.reshape(1, -1)

        # Use the trained model to predict the Ethereum open price for the specified future date
        predicted_open_price = regr.predict(input_features)[0]

        # Adjust the predicted price based on the previous prediction and actual open price
        if previous_prediction is not None:
            actual_open_price = y.loc[input_date]
            adjustment = actual_open_price - previous_prediction
            predicted_open_price += adjustment

        predicted_open_prices.append(predicted_open_price)
        previous_prediction = predicted_open_price
    except KeyError:
        # Handle the case where the date doesn't exist in the index
        print(f"Warning: Data for date {input_date} not found. Using the previous date's prediction.")
        if predicted_open_prices:
            # Use the last available prediction as an estimate
            predicted_open_prices.append(predicted_open_prices[-1])
        else:
            print("Error: No previous predictions available.")
            break

# Create a DataFrame to store the predicted open prices with corresponding dates
predicted_data = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Open": predicted_open_prices
})

# Display the predicted open prices
print(predicted_data)
