import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load the dataset
file_path = 'BMRI_JK_stock_data.csv'
stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Using 'Close' prices for modeling
data = stock_data[['Close']].dropna()

# Perform Dickey-Fuller test
def perform_dickey_fuller(series):
    result = adfuller(series)
    if result[1] > 0.05:
        return False
    return True

# Check for stationarity
if not perform_dickey_fuller(data['Close']):
    data_diff = data['Close'].diff().dropna()  # First differencing
else:
    data_diff = data['Close']

# Splitting the dataset into training and testing sets
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[:train_size], data_diff[train_size:]

# Building the ARIMA model with optimized order
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# Making predictions
y_pred_diff = model_fit.forecast(steps=len(test))
y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
y_test = data['Close'].iloc[train_size:]

# Ensure both arrays have the same length
min_len = min(len(y_test), len(y_pred))
y_test = y_test[:min_len]
y_pred = y_pred[:min_len]

# Metrics for evaluation
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit UI
st.title("Stock Price Prediction with ARIMA")

# Plotting actual vs predicted prices
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(data.index, data['Close'], color='blue', label='Actual Price')
ax.plot(test.index, y_pred, color='red', label='Predicted Price')

# Formatting the plot
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.set_title('Stock Price Prediction with ARIMA (Optimized)', fontsize=20)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.xticks(rotation=30)
plt.legend()

st.pyplot(fig)

# Displaying metrics
metrics = {
    'MAE': mae,
    'MAPE': mape,
    'MSE': mse,
    'RMSE': rmse
}

st.subheader("Model Evaluation Metrics")
for metric, value in metrics.items():
    st.write(f"{metric}: {value:.4f}")

# Displaying predictions in a list
st.subheader("Predicted Stock Prices")
predicted_prices = pd.DataFrame({
    'Date': test.index,
    'Predicted Price': y_pred
})
st.write(predicted_prices)
