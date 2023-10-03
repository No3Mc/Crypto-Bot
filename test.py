import pandas as pd
excel_file_path = 'eth.xlsx'
df_2020 = pd.read_excel(excel_file_path, header=0)
print(df_2020.head())

