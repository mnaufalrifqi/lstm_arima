import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def load_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

def check_stationarity(series):
    result = adfuller(series)
    return result[1] < 0.05  # Return True if stationary, False if not

def train_arima_model(train_data):
    model = ARIMA(train_data, order=(2,1,2))
    model_fit = model.fit()
    return model_fit

def preprocess_data_for_lstm(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close']])
    return data_scaled, scaler

def prepare_lstm_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    X = np.array(X).reshape((len(X), 1, look_back))
    y = np.array(y)
    return X, y

def main():
    st.title("Stock Price Prediction Web App")
    
    st.sidebar.header("Stock Data Input")
    stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., BMRI.JK):", "BMRI.JK")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-01"))
    
    data = load_stock_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.error("Stock data not available. Check the stock symbol.")
        return
    
    st.write("## Stock Data Preview")
    st.write(data.head())
    
    # Plot stock prices
    st.write("## Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (IDR)')
    ax.set_title(f'{stock_symbol} Closing Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30)
    plt.legend()
    st.pyplot(fig)
    
    # Select Model
    model_type = st.sidebar.selectbox("Select Prediction Model", ["ARIMA", "LSTM"])
    
    train_size = int(len(data) * 0.8)
    train, test = data['Close'][:train_size], data['Close'][train_size:]
    
    if model_type == "ARIMA":
        st.write("## ARIMA Model Training")
        
        if not check_stationarity(train):
            train = train.diff().dropna()
        
        model_fit = train_arima_model(train)
        
        predictions = model_fit.forecast(steps=len(test))
        predictions = data['Close'].iloc[train_size-1] + predictions.cumsum()
        
        st.write("### ARIMA Model Summary")
        st.text(model_fit.summary())
        
    else:
        st.write("## LSTM Model Training")
        
        data_scaled, scaler = preprocess_data_for_lstm(data)
        X_train, y_train = prepare_lstm_data(data_scaled[:train_size])
        X_test, y_test = prepare_lstm_data(data_scaled[train_size:])
        
        model = load_model("final_model_lstm.h5")
        y_pred_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(y_pred_scaled)
    
    # Plot Predictions
    st.write("## Stock Price Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
    ax.plot(data.index[train_size+1:], predictions, label='Predicted Price', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (IDR)')
    ax.set_title(f'{stock_symbol} Price Prediction - {model_type}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30)
    plt.legend()
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
