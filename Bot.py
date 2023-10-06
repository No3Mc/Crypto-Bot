import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


csv_file = ("Ethereum_History(2019-2023).csv")
data = pd.read_csv(csv_file)
train = data.drop(["Date"], axis = 1)
test = data["Open", "High", "Low", "Close"]
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=2)
regr = LinearRegression()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)
pred
regr.score
