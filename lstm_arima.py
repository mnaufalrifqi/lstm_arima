import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Streamlit App
st.title("Stock Price Prediction: ARIMA vs LSTM")

# Sidebar Inputs
st.sidebar.header("Data Selection")
stock_symbol = "BMRI.JK"
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-01"))

# Load stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data[['Close']].dropna()

# Sidebar Model Selection
st.sidebar.header("Select Model")
model_type = st.sidebar.selectbox("Prediction Model:", ["ARIMA", "LSTM"])

# Plot historical stock data
st.subheader("Historical Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Close Price', color='blue')
ax.set_title("Historical Stock Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (IDR)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend()
st.pyplot(fig)

# ARIMA Model
if model_type == "ARIMA":
    st.subheader("ARIMA Model Prediction")
    
    # Perform Dickey-Fuller Test to check for stationarity
    def check_stationarity(series):
        result = adfuller(series)
        return result[1] < 0.05  # Returns True if stationary
    
    is_stationary = check_stationarity(data['Close'])
    
    # Apply differencing to make the data stationary if needed
    if not is_stationary:
        data_diff = data['Close'].diff().dropna()
    else:
        data_diff = data['Close']
    
    # Splitting the dataset into training and testing sets
    train_size = int(len(data_diff) * 0.8)
    train, test = data_diff[:train_size], data_diff[train_size:]
    
    # Fit ARIMA Model with order (2, 1, 2)
    arima_model = ARIMA(train, order=(2, 1, 2))
    arima_fit = arima_model.fit()
    
    # Forecast
    y_pred_diff = arima_fit.forecast(steps=len(test))  # Predict the differenced values
    
    # Reverse differencing to get the actual price predictions
    last_price = data['Close'].iloc[train_size - 1]  # Last price in training set
    y_pred = last_price + np.cumsum(y_pred_diff)  # Add the cumulative sum of the differenced predictions to the last known price
    y_test = data['Close'].iloc[train_size:].values  # Actual test data (prices)
    
    # Cek apakah ada NaN atau Infinity
if np.any(np.isnan(y_test)) or np.any(np.isnan(y_pred)):
    st.warning("Data contains NaN values. Please check the data.")
    y_test = np.nan_to_num(y_test)  # Menggantikan NaN dengan 0 atau nilai lainnya
    y_pred = np.nan_to_num(y_pred)

if np.any(np.isinf(y_test)) or np.any(np.isinf(y_pred)):
    st.warning("Data contains infinity values. Please check the data.")
    y_test = np.where(np.isinf(y_test), 0, y_test)  # Gantikan inf dengan 0
    y_pred = np.where(np.isinf(y_pred), 0, y_pred)

# Pastikan data dalam format 1D
y_test = y_test.flatten()
y_pred = y_pred.flatten()

# Hitung MAE dan RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Tampilkan hasil evaluasi
st.write("Mean Absolute Error (MAE):", round(mae, 4))
st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))

