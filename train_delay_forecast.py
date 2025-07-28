import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('train_15104_delay_history.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Delay_Minutes'] = pd.to_numeric(df['Delay_Minutes'], errors='coerce')

imputer = IterativeImputer(max_iter=10, random_state=42)
df[['Delay_Minutes']] = imputer.fit_transform(df[['Delay_Minutes']])

df['Station'] = df['Station'].fillna(method='ffill')  

df.dropna(subset=['Date', 'Station'], inplace=True)

stations = df['Station'].drop_duplicates()
station_order = {station: idx for idx, station in enumerate(stations)}

forecast_steps = 5
all_forecasts = []

for station in stations:
    df_station = df[df['Station'] == station].copy()
    df_station.set_index('Date', inplace=True)
    df_station.sort_index(inplace=True)

    if len(df_station) < 10:
        continue

    try:
        model = ARIMA(df_station['Delay_Minutes'], order=(1, 1, 1))
        model_fit = model.fit()

        last_date = df_station.index.max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
        forecast_values = model_fit.forecast(steps=forecast_steps)

        for date, val in zip(forecast_dates, forecast_values):
            all_forecasts.append({
                'Station': station,
                'Date': date.strftime('%Y-%m-%d'),
                'Forecasted_Delay': round(val, 2)
            })
    except Exception as e:
        print(f"Error forecasting for station {station}: {e}")

forecast_df = pd.DataFrame(all_forecasts)
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
forecast_df['Route_Order'] = forecast_df['Station'].map(station_order)
forecast_df.sort_values(by=['Date', 'Route_Order'], inplace=True)
forecast_df.drop(columns=['Route_Order'], inplace=True)

print("\nForecasted Delay (Next {} Days):\n".format(forecast_steps))
print(forecast_df.to_string(index=False))

forecast_df.to_csv('station_delay_forecast.csv', index=False)
