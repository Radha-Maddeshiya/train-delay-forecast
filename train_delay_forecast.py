import pandas as pd

df = pd.read_csv('train_15104_delay_history.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Delay_Minutes'] = pd.to_numeric(df['Delay_Minutes'], errors='coerce')
df.dropna(subset=['Date', 'Delay_Minutes', 'Station'], inplace=True)

stations = df['Station'].unique()
forecast_steps = 5
all_forecasts = []

last_date = df['Date'].max()

for station in stations:
    station_df = df[df['Station'] == station]
    if len(station_df) < 2:
        continue  

    avg_delay = station_df['Delay_Minutes'].mean()

    for i in range(1, forecast_steps + 1):
        forecast_date = last_date + pd.Timedelta(days=i)
        all_forecasts.append({
            'Station': station,
            'Date': forecast_date.strftime('%Y-%m-%d'),
            'Forecasted_Delay': round(avg_delay, 2)
        })

forecast_df = pd.DataFrame(all_forecasts)

if forecast_df.empty:
    print("\n⚠️ Forecast data is empty — not enough records to generate output.")
else:
    station_order = df.drop_duplicates('Station')['Station'].tolist()
    forecast_df['Station'] = pd.Categorical(forecast_df['Station'], categories=station_order, ordered=True)
    forecast_df.sort_values(by=['Date', 'Station'], inplace=True)

    print("\n📊 Forecasted Delay using Average (Next {} Days):\n".format(forecast_steps))
    print(forecast_df.to_string(index=False))

    forecast_df.to_csv('station_delay_forecast.csv', index=False)
