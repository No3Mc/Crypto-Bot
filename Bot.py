import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


csv_file = ("Ethereum_History(2019-2020).csv")
data = pd.read_csv(csv_file)
print(data.tail())