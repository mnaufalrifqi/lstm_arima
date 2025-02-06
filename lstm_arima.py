import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Streamlit App
st.title("Stock Price Prediction: ARIMA")

# Sidebar Inputs
st.sidebar.header("Data Selection")
stock_symbol = "BMRI.JK"
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-01"))

# Load stock data from Yahoo Finance
data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data[['Close']].dropna()

# Sidebar Model Selection
st.sidebar.header("Select Model")
model_type = "ARIMA"  # ARIMA is selected here, as per your request

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

# ARIMA Model Implementation
if model_type == "ARIMA":
    st.subheader("ARIMA Model Prediction")
    
    # Perform Dickey-Fuller Test
    def check_stationarity(series):
        result = adfuller(series)
        return result[1] < 0.05  # Returns True if stationary
    
    # Check if the data is stationary
    is_stationary = check_stationarity(data['Close'])
    if not is_stationary:
        st.write("Data is not stationary. Performing differencing...")
        data_diff = data['Close'].diff().dropna()  # Differencing the data to make it stationary
        # Re-check stationarity after differencing
        result = adfuller(data_diff)
        st.write("Dickey-Fuller Test on Differenced Data:")
        st.write(f"Test Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value:.4f}")
    else:
        st.write("Data is already stationary.")
        data_diff = data['Close']  # Use original data if already stationary
    
    # Split the data into training and testing sets after differencing
    train_size = int(len(data_diff) * 0.8)
    train, test = data_diff[:train_size], data_diff[train_size:]
    
    # Build and fit ARIMA model
    model = ARIMA(train, order=(2, 1, 2))  # ARIMA(p,d,q) parameters
    model_fit = model.fit()
    
    # Make forecast
    y_pred_diff = model_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()  # Convert back to original price level
    y_test = data['Close'].iloc[train_size:]
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Display metrics
    st.write("Mean Absolute Error (MAE):", round(mae, 4))
    st.write("Mean Absolute Percentage Error (MAPE):", round(mape, 4))
    st.write("Mean Squared Error (MSE):", round(mse, 4))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 4))
    
    # Plot Predictions vs Actuals
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
