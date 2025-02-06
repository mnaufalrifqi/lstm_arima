import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set up the page layout
st.set_page_config(page_title="ARIMA Model Prediction", layout="wide")

# Title of the app
st.title('Stock Price Prediction using ARIMA')

# Load dataset
file_path = 'BMRI_JK_stock_data.csv'
stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data = stock_data[['Close']].dropna()

# Plot original data
st.subheader('Original Stock Price Data')
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['Close'], color='blue', label='Original Data')
ax1.set_title('Original Data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price')
ax1.legend()
st.pyplot(fig1)

# Perform Dickey-Fuller test
def perform_dickey_fuller(series):
    result = adfuller(series)
    return result[1]

# Check for stationarity
adf_test = adfuller(data['Close'])
p_value = adf_test[1]

if p_value > 0.05:
    st.write("Data is not stationary, performing differencing...")
    data_diff = data['Close'].diff().dropna()  # First differencing
else:
    st.write("Data is stationary.")
    data_diff = data['Close']

# Plot differenced data
st.subheader('Differenced Data')
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data_diff.index, data_diff, color='orange', label='Differenced Data')
ax2.set_title('Differenced Data')
ax2.set_xlabel('Date')
ax2.set_ylabel('Differenced Close Price')
ax2.legend()
st.pyplot(fig2)

# Splitting dataset
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[:train_size], data_diff[train_size:]

# Building ARIMA model
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# Making predictions
y_pred_diff = model_fit.forecast(steps=len(test))
y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
y_test = data['Close'].iloc[train_size:]

# Ensure both arrays have the same length
min_len = min(len(y_test), len(y_pred))
y_test = y_test[:min_len]
y_pred = y_pred[:min_len]

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Plotting actual vs predicted prices
st.subheader('Actual vs Predicted Prices')
fig3, ax3 = plt.subplots(figsize=(15, 7))
ax3.plot(data.index, data['Close'], color='blue', label='Actual Price')
ax3.plot(test.index, y_pred, color='red', label='Predicted Price')
ax3.set_title('Stock Price Prediction using ARIMA', fontsize=20)
ax3.set_xlabel('Time')
ax3.set_ylabel('Stock Price')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.xticks(rotation=30)
ax3.legend()
st.pyplot(fig3)

# Display metrics
metrics = {
    'MAE': mae,
    'MAPE': mape,
    'MSE': mse,
    'RMSE': rmse
}

st.subheader('Model Evaluation Metrics')
for metric, value in metrics.items():
    st.write(f"{metric}: {value:.4f}")


# User input for a specific prediction date
st.subheader('Prediksi pada Tanggal Tertentu')
prediction_date = st.date_input('Pilih Tanggal untuk Prediksi', min_value=test.index.min(), max_value=test.index.max(), value=test.index[0])

# Convert the selected prediction date to a pandas Timestamp object
prediction_date_ts = pd.Timestamp(prediction_date)

# Ensure the prediction_date_ts is in the range of test.index
if prediction_date_ts not in test.index:
    st.write(f"Tanggal {prediction_date} tidak ada dalam data.")
else:
    # Find the index for the selected date in the test set
    prediction_date_index = test.index.get_loc(prediction_date_ts)

    # Get the predicted value for the selected date
    predicted_value = y_pred[prediction_date_index]

    # Get the actual value for the selected date
    actual_value = y_test.iloc[prediction_date_index]

    # Determine if the stock price is "Naik" or "Turun"
    price_change = predicted_value - actual_value
    price_change_percentage = (price_change / actual_value) * 100
    trend = "Naik" if price_change > 0 else "Turun"

    # Display the prediction result
    st.write(f"Prediksi Harga Saham pada {prediction_date}: {predicted_value:.2f} IDR")
    st.write(f"Harga Aktual pada {prediction_date}: {actual_value:.2f} IDR")
    st.write(f"Perubahan Harga: {price_change:.2f} IDR ({price_change_percentage:.2f}%)")
    st.write(f"Tren: {trend}")
