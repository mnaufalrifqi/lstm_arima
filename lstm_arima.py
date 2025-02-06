import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import streamlit as st

# Load the dataset
file_path = 'BMRI_JK_stock_data.csv'
stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Using 'Close' prices for modeling
data = stock_data[['Close']].dropna()

# Streamlit display of the original data
st.title('ARIMA Stock Price Prediction')
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
