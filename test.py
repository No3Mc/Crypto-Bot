import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
csv_file = "Updated-Ethereum History(2019-2023) - Ethereum HIstory(2019-2023).csv"
data = pd.read_csv(csv_file)

# Preprocess the data: Set "Date" as the index and ensure it's in datetime format
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data.set_index('Date', inplace=True)

# Define features (X) and target variable (y)
X = data[["High", "Low", "Close"]]  # Features (excluding "Open" as it's the target variable)
y = data["Open"]  # Target variable

# Initialize and train the linear regression model on the entire dataset
regr = LinearRegression()
regr.fit(X, y)

# Define a range of future dates for prediction (replace with your desired date range)
future_dates = pd.date_range(start="01/01/2023", end="01/10/2023")  # Example date range

# Make predictions for each future date
predicted_open_prices = []

for input_date in future_dates:
    # Create a feature vector for prediction
    input_features = X.loc[input_date][["High", "Low", "Close"]].values.reshape(1, -1)

    # Use the trained model to predict the Ethereum open price for the specified future date
    predicted_open_price = regr.predict(input_features)[0]
    predicted_open_prices.append(predicted_open_price)

# Create a DataFrame to store the predicted open prices with corresponding dates
predicted_data = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Open": predicted_open_prices
})

# Display the predicted open prices
print(predicted_data)
