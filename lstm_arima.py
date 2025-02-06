import streamlit as st 
import yfinance as yf 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import load_model 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
import math 
import plotly.graph_objects as go  
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Streamlit app
def main():
    st.title("Stock Price Prediction: ARIMA vs LSTM")

    # Sidebar for data download
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., BBCA.JK):", "BBCA.JK")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Data preprocessing
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Sidebar for model selection
    st.sidebar.header("Select Model")
    model_type = st.sidebar.selectbox("Select Model Type:", ["LSTM", "ARIMA"])

    if model_type == "LSTM":
        model = load_model("final_model_lstm.h5")
        X, y = prepare_lstm_data(scaled_data, 120)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
    else:
        train_size = int(len(close_prices) * 0.8)
        train, test = close_prices[:train_size], close_prices[train_size:]
        model = ARIMA(train, order=(2,1,2))
        model_fit = model.fit()
        y_pred_diff = model_fit.forecast(steps=len(test))
        y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()

    y_test_orig = data['Close'].iloc[train_size:]

    # Calculate metrics
    mse = mean_squared_error(y_test_orig[:len(y_pred)], y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_orig[:len(y_pred)], y_pred)
    mae = mean_absolute_error(y_test_orig[:len(y_pred)], y_pred)

    # Display results
    st.header(f"Results for {model_type} Model")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("Mean Absolute Percentage Error (MAPE):", mape)
    st.write("Mean Absolute Error (MAE):", mae)

    # Visualize predictions
    st.header("Visualize Predictions")
    visualize_predictions(data, train_size, y_test_orig, y_pred)

def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def visualize_predictions(data, train_size, y_test_orig, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_test_orig[:len(y_pred)], mode='lines', name="Actual Stock Prices", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_pred, mode='lines', name="Predicted Stock Prices", line=dict(color='red')))
    fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Stock Price (IDR)", template='plotly_dark')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
