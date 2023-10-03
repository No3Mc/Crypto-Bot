import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_excel("eth.xlsx", engine="openpyxl", sheet_name="ETHUSD")

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

df.dropna(subset=['Date'], inplace=True)

df.set_index('Date', inplace=True)

df_2020 = df[df.index.year == 2020]
df_2021 = df[df.index.year == 2021]

if 'Close' in df_2020.columns:
    X = df_2020[['Close']].values
    y = df_2021[['Close']].values

    model = LinearRegression()

    model.fit(X, y)


    last_2020_price = df_2020.iloc[-1]['Close']
    price_2021_prediction = model.predict([[last_2020_price]])

    print(f"Predicted price for 2021: {price_2021_prediction[0][0]:.2f}")
else:
    print("The 'Close' column does not exist in df_2020.")
