import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('T1.csv', parse_dates=['Date/Time'], dayfirst=True)

df.set_index('Date/Time', inplace=True)


df.rename(columns={
    'LV ActivePower (kW)': 'LV_ActivePower',
    'Wind Speed (m/s)': 'WindSpeed',
    'Theoretical_Power_Curve (KWh)': 'TheoreticalPower',
    'Wind Direction (°)': 'WindDirection'
}, inplace=True)


plt.figure(figsize=(6,4))
sns.scatterplot(x='WindSpeed', y='LV_ActivePower', data=df)
plt.title("Wind Speed vs LV ActivePower")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("LV ActivePower (kW)")
plt.show()


df['PerformanceScore'] = (df['LV_ActivePower'] / df['TheoreticalPower']) * 100

df['PerformanceCategory'] = pd.cut(
    df['PerformanceScore'],
    bins=[0, 80, 95, 100],
    labels=['Poor', 'Moderate', 'Good']
)

print(df[['LV_ActivePower', 'TheoreticalPower', 'PerformanceScore', 'PerformanceCategory']].head())

df['LV_ActivePower'].fillna(method='ffill', inplace=True)


def fn(series, window_size=5):
  
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)


X, y = fn(df['LV_ActivePower'].values, window_size=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = LinearRegression()
model.fit(X_train, y_train)


preds = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual')
plt.plot(preds, label='Predicted')
plt.title("LV ActivePower Forecast")
plt.xlabel("Time")
plt.ylabel("LV ActivePower (kW)")
plt.legend()
plt.show()

df['Anomaly'] = df['PerformanceScore'] < 80


def categorize(score):
    if score >= 90:
        return "Good"
    elif score >= 75:
        return "Moderate"
    else:
        return "Poor"

df['Category'] = df['PerformanceScore'].apply(categorize)


print(df[df['Anomaly']].head())


