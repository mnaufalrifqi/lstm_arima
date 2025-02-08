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

# Display initial Close price
st.subheader("Initial Close Price (BMRI.JK)")

# Create the plot
plt.figure(figsize=(15, 7))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.plot(data.index, data['Close'], label='Close')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Price (Rp)')
plt.title("Harga saham BMRI", fontsize=20)
plt.legend()

# Auto-format the x-axis dates
plt.gcf().autofmt_xdate()

# Show the plot in Streamlit
st.pyplot(plt)


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

# Show model summary in Streamlit
st.subheader("LSTM Model Summary")
model.summary(print_fn=lambda x: st.text(x))

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=10)])

# Visualize Loss and MAE during training
fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_title('Training and Validation Loss')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss (MSE)')
ax_loss.legend()
st.pyplot(fig_loss)

fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
ax_mae.plot(history.history['mae'], label='Training MAE')
ax_mae.plot(history.history['val_mae'], label='Validation MAE')
ax_mae.set_title('Training and Validation Mean Absolute Error (MAE)')
ax_mae.set_xlabel('Epochs')
ax_mae.set_ylabel('MAE')
ax_mae.legend()
st.pyplot(fig_mae)

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

# Display metrics in Streamlit
st.subheader("Model Evaluation Metrics")
st.write(f"MAE: {mae:.2f}")
st.write(f"MAPE: {mape:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# Visualize the comparison of actual vs predicted stock prices
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
        data_diff = data

    # ARIMA model fitting
    st.subheader("Fitting ARIMA Model")

    # Auto ARIMA model
    auto_arima_model = auto_arima(data_diff, seasonal=False, trace=True)
    st.write("Optimal ARIMA Parameters:")
    st.write(auto_arima_model.summary())

    # Forecasting using ARIMA
    forecast_steps = 30
    forecast = auto_arima_model.predict(n_periods=forecast_steps)

    # Display ARIMA forecasted values
    st.subheader(f"ARIMA Forecasted Prices for Next {forecast_steps} Days")
    forecast_dates = pd.date_range(data.index[-1], periods=forecast_steps+1, freq='B')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': forecast})

    st.write(forecast_df)

    # Visualize ARIMA forecast
    st.subheader("ARIMA Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], color='blue', label='Original Data')
    ax.plot(forecast_df['Date'], forecast_df['Predicted Close'], color='red', label='Forecasted Data')
    ax.set_title('ARIMA Model Forecasting')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

    # Display ARIMA results table
    st.subheader("ARIMA Forecast Results")
    st.write(forecast_df)
