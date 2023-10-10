import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


excel_file_path = 'eth.xlsx'
df_2020 = pd.read_excel(excel_file_path, header=None).T

df_2020.columns = df_2020.iloc[0]
df_2020 = df_2020.iloc[1:]
df_2020.columns = ['Date', 'Open', 'High', 'Low', 'Close']
df_2020['Date'] = pd.to_datetime(df_2020['Date'], format='%m/%d/%Y')

training_data = df_2020[df_2020['Date'] <= '2021-01-01']
testing_data = df_2020[df_2020['Date'] > '2021-01-01']

time_series = training_data['Close'].astype(float)

model = ARIMA(time_series, order=(5, 1, 0))
model_fit = model.fit()

n_predictions = len(testing_data)
predictions = model_fit.forecast(steps=n_predictions)

predictions_index = pd.date_range(start=testing_data['Date'].iloc[0], periods=n_predictions)

predicted_df = pd.DataFrame({'Date': predictions_index, 'Predicted_Close': predictions})

print(predicted_df)

