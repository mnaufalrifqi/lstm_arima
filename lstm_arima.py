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
from statsmodels.tsa.stattools import adfuller, acf, pacf
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
    
    def perform_dickey_fuller(series):
        result = adfuller(series)
        return result[1] < 0.05  # Returns True if stationary
    
    is_stationary = perform_dickey_fuller(data['Close'])
    data_diff = data['Close'].diff().dropna() if not is_stationary else data['Close']
    
    train_size = int(len(data_diff) * 0.8)
    train, test = data_diff[:train_size], data_diff[train_size:]
    
    try:
        with open("arima_model_3.pkl", "rb") as f:
            model_fit = pickle.load(f)
    except:
        model = ARIMA(train, order=(2,1,2))
        model_fit = model.fit()
        with open("arima_model_3.pkl", "wb") as f:
            pickle.dump(model_fit, f)
    
    y_pred_diff = model_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
    y_test = data['Close'].iloc[train_size:]
    
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.write("Mean Absolute Error (MAE):", round(mae, 4))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))
    
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
    
    scaler = MinMaxScaler()
    data['Close_scaled'] = scaler.fit_transform(data[['Close']])
    
    train_size = int(len(data) * 0.8)
    train, test = data['Close_scaled'][:train_size], data['Close_scaled'][train_size:]
    
    def create_sequences(data, look_back=1):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)
    
    look_back = 60
    X_train, y_train = create_sequences(train.values, look_back)
    X_test, y_test = create_sequences(test.values, look_back)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    try:
        lstm_model = load_model("lstm_model.h5")
    except:
        lstm_model = Sequential([
            LSTM(128, input_shape=(look_back, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        lstm_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
        lstm_model.save("lstm_model.h5")
    
    y_pred = lstm_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.write("Mean Absolute Error (MAE):", round(mae, 4))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
    ax.plot(test.index[look_back:], y_pred, label='Predicted Price (LSTM)', color='red')
    ax.set_title("Stock Price Prediction - LSTM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (IDR)")
    ax.legend()
    st.pyplot(fig)
