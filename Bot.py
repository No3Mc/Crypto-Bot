import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
csv_file = "Updated-Ethereum History(2019-2023).csv"
data = pd.read_csv(csv_file)

# Preprocess the data: Set "Date" as the index and ensure it's in datetime format
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data.set_index('Date', inplace=True)

# Ensure the datetime index is sorted in ascending order (if not already)
data.sort_index(inplace=True)

# Define features (X) and target variable (y)
X = data[["High", "Low", "Open"]]  # Features (excluding "Close" as it's the target variable)
y = data["Close"]  # Target variable

# Allow the user to input a specific future date
input_date = input("Enter a future date (m/d/yyyy): ")
try:
    input_date = pd.to_datetime(input_date, format="%m/%d/%Y")
except ValueError:
    print("Invalid date format. Please use the format 'm/d/yyyy'.")
    exit()

# Check if the specified date is within the range of available dates
if input_date < data.index.min() or input_date > data.index.max():
    print("Invalid date. Please enter a date within the range of available data.")
else:
    # Initialize and train the linear regression model on the entire dataset
    regr = LinearRegression()
    regr.fit(X, y)

    # Initialize a list for predicted close prices
    predicted_close_prices = []

    # Create a feature vector for prediction
    input_features = X.loc[input_date][["High", "Low", "Open"]].values.reshape(1, -1)

    # Use the trained model to predict the Ethereum close price for the specified future date
    predicted_close_price = regr.predict(input_features)[0]
    predicted_close_prices.append(predicted_close_price)

    # Create a DataFrame to store the predicted close price with the user-specified date
    predicted_data = pd.DataFrame({
        "Date": [input_date],
        "Predicted_Close": predicted_close_prices
    })

    # Display the predicted close price
    print(predicted_data)

    # You can further evaluate the model's performance if needed
    # (e.g., calculate MAE on the entire dataset), but it's not necessary for this specific prediction.

