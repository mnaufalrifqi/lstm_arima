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

# Perform Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    return result[1] < 0.05  # Returns True if stationary

st.subheader("Uji Stasioneritas")
if not check_stationarity(data['Close']):
    data_diff = data['Close'].diff().dropna()
    st.write("Data tidak stasioner, melakukan differencing...")
else:
    data_diff = data['Close']
    st.write("Data sudah stasioner.")

# Plot differenced data
st.subheader("Plot Data Setelah Differencing")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_diff, color='orange', label='Differenced Data')
ax.set_title("Differenced Data")
ax.set_xlabel("Date")
ax.set_ylabel("Differenced Close Price")
ax.legend()
st.pyplot(fig)

# Splitting the dataset
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[:train_size], data_diff[train_size:]

# Display ACF & PACF
st.subheader("ACF & PACF")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(train, lags=20, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")
plot_pacf(train, lags=20, ax=axes[1], method='ywm')
axes[1].set_title("Partial Autocorrelation Function (PACF)")
st.pyplot(fig)

# Train ARIMA model
st.subheader("Model ARIMA")
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
st.text(model_fit.summary())

# Forecasting
y_pred_diff = model_fit.forecast(steps=len(test))
y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
y_test = data['Close'].iloc[train_size:]

# Ensure both arrays have the same length
y_test, y_pred = y_test[:len(y_pred)], y_pred[:len(y_test)]

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("Evaluasi Model")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot predictions
# Plot hasil prediksi
st.subheader("Grafik Prediksi Harga Saham")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Close"], label="Harga Aktual", color="blue")
ax.plot(test.index, y_pred, label="Harga Prediksi", color="red", linestyle="dashed")
ax.set_title("Prediksi Harga Saham dengan ARIMA")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga Saham")
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
