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

    # Splitting the dataset into training and testing sets
    train_size = int(len(data_diff) * 0.8)
    train, test = data_diff[:train_size], data_diff[train_size:]
    st.write("Train Data Sample:", train.head(10))
    st.write("Test Data Sample:", test.head(10))

    # Calculate and display ACF and PACF values
    acf_values = acf(train, nlags=20)
    pacf_values = pacf(train, nlags=20, method='ywm')
    st.write("ACF Values:")
    for i, v in enumerate(acf_values):
        st.write(f"Lag {i}: {v:.4f}")

    st.write("\nPACF Values:")
    for i, v in enumerate(pacf_values):
        st.write(f"Lag {i}: {v:.4f}")

    # Plot ACF and PACF with lag values
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(train, lags=20, ax=axes[0])
    axes[0].set_title('ACF (Autocorrelation Function)')

    plot_pacf(train, lags=20, ax=axes[1], method='ywm')
    axes[1].set_title('PACF (Partial Autocorrelation Function)')

    st.pyplot(fig)

    # Building the ARIMA model with optimized order
    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()
    st.write(model_fit.summary())

    # Making predictions
    y_pred_diff = model_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
    y_test = data['Close'].iloc[train_size:]
    st.write("Predicted Prices:", y_pred)

    # Ensure both arrays have the same length
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]

    # Metrics for evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write("Mean Absolute Error (MAE):", round(mae, 4))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))
    
    # Plotting actual vs predicted prices
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data.index, data['Close'], color='blue', label='Harga Aktual')
    ax.plot(test.index, y_pred, color='red', label='Harga Prediksi')

    # Formatting the plot
    ax.set_xlabel('Waktu')
    ax.set_ylabel('Harga Saham')
    ax.set_title('Prediksi Harga Saham dengan ARIMA (Optimized)', fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.xticks(rotation=30)
    ax.legend()

    st.pyplot(fig)

    # Displaying metrics
    metrics = {
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse,
        'RMSE': rmse
    }

    # Output metrics
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")

    

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
