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
    
    # Perform Dickey-Fuller Test
    def check_stationarity(series):
        result = adfuller(series)
        return result[1] < 0.05  # Returns True if stationary
    
    is_stationary = check_stationarity(data['Close'])
    if not is_stationary:
        data_diff = data['Close'].diff().dropna()
    else:
        data_diff = data['Close']
    
    # Plot differenced data
    st.subheader("Differenced Data")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_diff.index, data_diff, label='Differenced Data', color='orange')
    ax.set_title("Differenced Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Differenced Close Price")
    ax.legend()
    st.pyplot(fig)
    
    # ACF and PACF plots
    st.subheader("ACF and PACF Plots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(data_diff, lags=20, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")
    plot_pacf(data_diff, lags=20, ax=ax[1])
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Fit ARIMA Model
    arima_model = ARIMA(train, order=(2,1,2))
    arima_fit = arima_model.fit()
    
    st.text("ARIMA Model Summary:")
    st.text(arima_fit.summary())
    
    # Forecast
    y_pred_diff = arima_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
    y_test = data['Close'].iloc[train_size:]
        # Plot Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
    ax.plot(test.index, y_pred, label='Predicted Price (ARIMA)', color='red')
    ax.set_title("Stock Price Prediction - ARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (IDR)")
    ax.legend()
    st.pyplot(fig)
    

# LSTM Model
elif model_type == "LSTM":
    st.subheader("LSTM Model Prediction")
    
    # Normalize Data
    scaler = MinMaxScaler()
    data['Close_scaled'] = scaler.fit_transform(data[['Close']])
    
    # Train LSTM Model
    lstm_model = Sequential([
        LSTM(128, input_shape=(60, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Load existing model if available
    try:
        lstm_model.load_weights("lstm_model.h5")
    except:
        pass
    
    # Predict future values
    X_test = data['Close_scaled'].values[-60:].reshape(1, 60, 1)
    predicted_price = scaler.inverse_transform(lstm_model.predict(X_test))[0][0]
    actual_price = data['Close'].iloc[-1]
    trend = "Up" if predicted_price > actual_price else "Down"
    
    st.write(f"Predicted Price on {prediction_date}: {predicted_price:.2f} IDR ({trend})")
