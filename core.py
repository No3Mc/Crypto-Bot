import pandas as pd


excel_file_path = 'eth.xlsx'
df_2020 = pd.read_excel(excel_file_path, header=None).T


df_2020.columns = df_2020.iloc[0]

df_2020 = df_2020.iloc[1:]


df_2020.columns = ['Date', 'Open', 'High', 'Low', 'Close']


df_2020['Date'] = pd.to_datetime(df_2020['Date'], format='%m/%d/%Y')


training_start_date = '2020-01-01'
training_end_date = '2021-01-01'
testing_start_date = '2021-01-02'
testing_end_date = '2022-01-01'

training_data = df_2020[(df_2020['Date'] >= training_start_date) & (df_2020['Date'] <= training_end_date)]
testing_data = df_2020[(df_2020['Date'] >= testing_start_date) & (df_2020['Date'] <= testing_end_date)]


print("Training Data:")
print(training_data.head())

print("\nTesting Data:")
print(testing_data.head())
