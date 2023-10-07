import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
csv_file = "Updated-Ethereum History(2019-2023) - Ethereum HIstory(2019-2023).csv"
data = pd.read_csv(csv_file)

# Preprocess the data: Set "Date" as the index and ensure it's in datetime format
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data.set_index('Date', inplace=True)

# Ensure the datetime index is sorted in ascending order (if not already)
data.sort_index(inplace=True)

# Define features (X) and target variable (y)
X = data[["High", "Low", "Open"]]  # Features (excluding "Close" as it's the target variable)
y = data["Close"]  # Target variable

# Check if the specified date range is valid
start_date = "01/01/2019"
end_date = "01/04/2023"
if start_date not in data.index or end_date not in data.index:
    print("Invalid date range. Please ensure the dates are present in the dataset.")
else:
    # Slice the dataset based on the specified date range
    train_data = data.loc[start_date:end_date]

    # Initialize and train the linear regression model on the training data
    regr = LinearRegression()
    regr.fit(X.loc[train_data.index], y.loc[train_data.index])

    # Define a range of future dates for prediction in "m/d/y" format
    future_dates = pd.date_range(start="01/05/2023", end="01/10/2023")  # Example date range

    # Initialize a list for predicted close prices
    predicted_close_prices = []

    for input_date in future_dates:
        try:
            # Create a feature vector for prediction
            input_features = X.loc[input_date][["High", "Low", "Open"]].values.reshape(1, -1)

            # Use the trained model to predict the Ethereum close price for the specified future date
            predicted_close_price = regr.predict(input_features)[0]
            predicted_close_prices.append(predicted_close_price)
        except KeyError:
            # Handle the case where the date doesn't exist in the index
            print(f"Warning: Data for date {input_date} not found. Using the previous date's prediction.")
            if predicted_close_prices:
                predicted_close_prices.append(predicted_close_prices[-1])
            else:
                print("Error: No previous predictions available.")
                break

    if predicted_close_prices:
        # Create a DataFrame to store the predicted close prices with corresponding dates
        predicted_data = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": predicted_close_prices
        })

        # Display the predicted close prices
        print(predicted_data)

        # Evaluate the model's accuracy on the training data
        y_true = y.loc[train_data.index]
        y_pred = regr.predict(X.loc[train_data.index])
        mae = mean_absolute_error(y_true, y_pred)
        print("Mean Absolute Error on Training Set:", mae)

