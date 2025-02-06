import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Streamlit UI for model selection
st.title("Stock Price Prediction: ARIMA vs LSTM")

model_choice = st.selectbox('Choose a model for prediction:', ['ARIMA', 'LSTM'])

if model_choice == 'LSTM':
    st.subheader("LSTM Model")

    # Download stock data
    data = yf.download("BMRI.JK", start="2019-12-01", end="2024-12-01")

    # Data processing
    ms = MinMaxScaler()
    data['Close_ms'] = ms.fit_transform(data[['Close']])

    # Split data into training and testing
    def split_data(data, train_size):
        size = int(len(data) * train_size)
        train, test = data.iloc[0:size], data.iloc[size:]
        return train, test

    train, test = split_data(data['Close_ms'], 0.8)

    # Split into X and Y
    def split_target(data, look_back=1):
        X, y = [], []
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back)]
            X.append(a)
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    X_train, y_train = split_target(train.values.reshape(len(train), 1))
    X_test, y_test = split_target(test.values.reshape(len(test), 1))

    # Reshape X for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define LSTM model
    model = Sequential([
        LSTM(128, input_shape=(1, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=10)])

    # Make predictions
    pred = model.predict(X_test)
    y_pred = np.array(pred).reshape(-1)

    # Inverse transform to get original scale
    y_pred_original = ms.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate percentage changes and directions
    percentage_changes = []
    directions = []
    for i in range(1, len(y_pred_original)):
        prev = y_pred_original[i - 1]
        curr = y_pred_original[i]
        change = ((curr - prev) / prev) * 100
        percentage_changes.append(change)
        directions.append("Naik" if change > 0 else "Turun")

    # Sync data
    min_length = min(len(test.index[1:]), len(y_pred_original[1:]), len(percentage_changes))
    sync_tanggal = test.index[1:][:min_length]
    sync_harga_prediksi = y_pred_original[1:][:min_length]
    sync_percentage_changes = percentage_changes[:min_length]
    sync_directions = directions[:min_length]

    # Prepare results for display
    predictions_df = pd.DataFrame({
        'Tanggal': sync_tanggal,
        'Harga Prediksi': sync_harga_prediksi,
        'Persentase Perubahan': sync_percentage_changes,
        'Tren': sync_directions
    })

    # Display metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.subheader("Model Evaluation Metrics")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MAPE: {mape:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    # Visualize the comparison of actual vs predicted stock prices in Streamlit
    fig = plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['Close'], color='blue', label='Harga Aktual')  # Plot actual prices (entire data)
    plt.plot(test.index[:-1], y_pred_original, color='red', label='Harga Prediksi')  # Plot predicted prices (test data)

    # Set labels and title
    plt.xlabel('Waktu')
    plt.ylabel('Harga Saham')
    plt.title('Prediksi Harga Saham BMRI LSTM', fontsize=20)

    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format on x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))    # Show label every 12 months

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30)

    # Add legend to distinguish between actual and predicted lines
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Display prediction results in a table
    st.subheader("Predicted Stock Prices with Change Direction")
    st.write(predictions_df)

elif model_choice == 'ARIMA':
    st.subheader("ARIMA Model")

    # Load the dataset
    file_path = 'BMRI_JK_stock_data.csv'
    stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Using 'Close' prices for modeling
    data = stock_data[['Close']].dropna()

    # Streamlit display of the original data
    st.subheader('Original Data')

    # Plot original data in Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], color='blue', label='Original Data')
    ax.set_title('Original Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

    # Perform Dickey-Fuller test
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

    # Check for stationarity and apply differencing if necessary
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

    # Building the ARIMA model with optimized order
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    st.write(model_fit.summary())

    # Making predictions
    y_pred_diff = model_fit.forecast(steps=len(test))
    y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
    y_test = data['Close'].iloc[train_size:]

    # Ensure both arrays have the same length
    min_len = min(len(y_test), len(y_pred))
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]

    # Metrics for evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Displaying metrics in Streamlit
    metrics = {
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse,
        'RMSE': rmse
    }

    st.subheader('Evaluation Metrics')
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")

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

    # Displaying the plot in Streamlit
    st.pyplot(fig)

    # Displaying predictions in a table
    st.subheader("Predicted Stock Prices with Change Direction")
    predicted_prices = pd.DataFrame({
        'Date': test.index,
        'Predicted Price': y_pred
    })

    # Adding a column for price change (up or down)
    predicted_prices['Price Change'] = predicted_prices['Predicted Price'].diff().apply(lambda x: 'naik' if x > 0 else 'turun')

    # Displaying the table with the new column
    st.write(predicted_prices)
