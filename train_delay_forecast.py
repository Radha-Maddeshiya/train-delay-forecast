import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('train_15104_delay_history.csv')

df['Delay_Minutes'] = df['Delay'].str.extract(r'(\d+)').astype(float)

last_station = df['Station'].iloc[-1]
df_station = df[df['Station'] == last_station]

df_station['Date'] = pd.to_datetime(df_station['Date'])
df_station.set_index('Date', inplace=True)

delay_series = df_station['Delay_Minutes'].dropna()

model = ARIMA(delay_series, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=5)

print("Forecasted Delay (in minutes) for next 5 days:")
print(forecast)

plt.figure(figsize=(10,5))
plt.plot(delay_series, label='Past Delay')
plt.plot(pd.date_range(start=delay_series.index[-1], periods=6, freq='D')[1:], forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Delay (minutes)')
plt.title('Train 15104 Delay Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
