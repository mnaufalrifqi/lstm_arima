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

# ARIMA Model
if model_type == "ARIMA":
    st.subheader("ARIMA Model Prediction")
    
    # Perform Dickey-Fuller Test
    def perform_dickey_fuller(series):
        result = adfuller(series)
        st.write("Dickey-Fuller Test Results:")
        st.write(f"Test Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value:.4f}")
        if result[1] > 0.05:
            st.write("The data is not stationary.")
        else:
            st.write("The data is stationary.")

    perform_dickey_fuller(data['Close'])

    # Check for stationarity
    adf_test = adfuller(data['Close'])
    p_value = adf_test[1]

    if p_value > 0.05:
        st.write("Data tidak stasioner, melakukan differencing...")
        data_diff = data['Close'].diff().dropna()  # First differencing
        perform_dickey_fuller(data_diff)
    else:
        st.write("Data sudah stasioner.")
        data_diff = data['Close']

    # Plot differenced data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_diff.index, data_diff, color='orange', label='Differenced Data')
    ax.set_title('Differenced Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Differenced Close Price')
    ax.legend()
    st.pyplot(fig)

      # Prediction and Evaluation for ARIMA
    if model_type == "ARIMA":
        y_pred_arima = arima_model_fit.forecast(steps=len(test))
        y_test_arima = data['Close'].iloc[train_size:]
        
        # Calculate ARIMA model metrics
        mae_arima = mean_absolute_error(y_test_arima, y_pred_arima)
        mape_arima = mean_absolute_percentage_error(y_test_arima, y_pred_arima)
        mse_arima = mean_squared_error(y_test_arima, y_pred_arima)
        rmse_arima = np.sqrt(mse_arima)

        # Display ARIMA model results
        st.header(f"ARIMA Model Results")
        st.write("Mean Absolute Error (MAE):", mae_arima)
        st.write("Mean Absolute Percentage Error (MAPE):", mape_arima)
        st.write("Mean Squared Error (MSE):", mse_arima)
        st.write("Root Mean Squared Error (RMSE):", rmse_arima)

        # Visualize ARIMA predictions
        st.header("Visualize ARIMA Predictions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_test_arima, mode='lines', name="Actual Stock Prices", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_pred_arima, mode='lines', name="Predicted Stock Prices", line=dict(color='red')))
        fig.update_layout(title="ARIMA Stock Price Prediction", xaxis_title="Date", yaxis_title="Stock Price", template='plotly_dark')
        st.plotly_chart(fig)

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
