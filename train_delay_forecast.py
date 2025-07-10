import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('train_15104_delay_history.csv')

df.dropna(how='all', inplace=True)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Delay_Minutes'] = pd.to_numeric(df['Delay_Minutes'], errors='coerce')

df.dropna(subset=['Date', 'Delay_Minutes'], inplace=True)

dates = df['Date'].tolist()
delays = df['Delay_Minutes'].tolist()

df_model = df.copy()
df_model.set_index('Date', inplace=True)

model = ARIMA(df_model['Delay_Minutes'], order=(1, 1, 1))
model_fit = model.fit()

forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)
last_date = df['Date'].max()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

plt.figure(figsize=(12,6))
plt.plot(dates, delays, marker='o', color='blue', label='Actual Delay')
plt.plot(forecast_dates, forecast, marker='o', linestyle='--', color='red', label='Forecast')

for i, val in enumerate(delays):
    plt.text(dates[i], val + 2, f"{val:.1f}", ha='center', color='blue', fontsize=8)

for i, val in enumerate(forecast):
    plt.text(forecast_dates[i], val + 2, f"{val:.1f}", ha='center', color='red', fontsize=8)

plt.title("Train 15104 Delay Forecast", fontsize=14, weight='bold')
plt.xlabel("Date")
plt.ylabel("Delay (minutes)")

ax = plt.gca()
all_dates = dates + list(forecast_dates)
ax.set_xticks(all_dates)
ax.set_xticklabels([d.strftime('%d-%b') for d in all_dates], rotation=45)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

plt.legend()
plt.tight_layout()
plt.show()
