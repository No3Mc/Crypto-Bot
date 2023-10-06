import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data
csv_file = "Updated-Ethereum History(2019-2023) - Ethereum HIstory(2019-2023).csv"
data = pd.read_csv(csv_file)

# Define features (X) and target (y)
X = data[["Open", "Close", "High", "Low"]]
y = data["Close"]  # You can choose "Close" as the target for prediction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Initialize and train the linear regression model
regr = LinearRegression()
regr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regr.predict(X_test)

# Calculate Mean Absolute Error (MAE) as an evaluation metric
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Optionally, you can also calculate R-squared (coefficient of determination)
r_squared = regr.score(X_test, y_test)
print("R-squared:", r_squared)

