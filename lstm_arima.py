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
    
    train_size = int(len(data_diff) * 0.8)
    train, test = data_diff[:train_size], data_diff[train_size:]
    
    # Fit ARIMA Model
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    
    # Forecast
    y_pred_diff = model_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
    y_test = data['Close'].iloc[train_size:]
    
    # Check for NaN or Inf values in y_pred and y_test
    if y_pred.isnull().any() or np.isnan(y_pred).any():
        st.write("Warning: Predicted values contain NaN or Inf.")
    if y_test.isnull().any() or np.isnan(y_test).any():
        st.write("Warning: Actual values contain NaN or Inf.")
    
    # Ensure the same length for y_test and y_pred
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    
    # Metrics for evaluation
    try:
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.write("Mean Absolute Error (MAE):", round(mae, 4))
        st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))
    except Exception as e:
        st.write(f"Error calculating metrics: {e}")

    # Check for NaN or Inf values in y_pred and y_test
if y_pred.isnull().any() or np.isnan(y_pred).any():
    st.write("Warning: Predicted values contain NaN or Inf.")
if y_test.isnull().any() or np.isnan(y_test.values).any():
    st.write("Warning: Actual values contain NaN or Inf.")

    
    # Plot Predictions
    def plot_predictions():
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
        ax.plot(test.index, y_pred, label='Predicted Price (ARIMA)', color='red')
        ax.set_title("Stock Price Prediction - ARIMA")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (IDR)")
        ax.legend()
        st.pyplot(fig)

    plot_predictions()

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
