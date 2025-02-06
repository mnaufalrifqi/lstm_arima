import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Fungsi untuk memuat model ARIMA
def load_arima_model():
    with open("arima_model_3.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Fungsi untuk memuat model LSTM
def load_lstm_model():
    model = load_model("lstm_model.h5")
    return model

# Fungsi untuk mengambil data saham
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Streamlit UI
st.title("Prediksi Harga Saham BMRI dengan ARIMA dan LSTM")

# Input user
start_date = st.date_input("Pilih Tanggal Mulai", value=pd.to_datetime("2019-12-01"))
end_date = st.date_input("Pilih Tanggal Akhir", value=pd.to_datetime("2024-12-01"))

data_load_state = st.text("Mengambil data...")
data = get_stock_data("BMRI.JK", start_date, end_date)
data_load_state.text("Data berhasil dimuat!")

# Plot harga saham
st.subheader("Harga Saham BMRI")
st.line_chart(data[['Close']])

# Memuat model ARIMA
arima_model = load_arima_model()

# Prediksi dengan ARIMA
st.subheader("Prediksi Harga Saham dengan ARIMA")
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]
y_pred_arima = arima_model.forecast(steps=len(test))

y_test = data['Close'][train_size:]
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Harga Aktual')
plt.plot(test.index, y_pred_arima, label='Prediksi ARIMA', color='red')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham')
plt.title('Prediksi Harga Saham dengan ARIMA')
plt.legend()
st.pyplot(plt)

# Evaluasi ARIMA
mae_arima = mean_absolute_error(y_test, y_pred_arima)
mape_arima = mean_absolute_percentage_error(y_test, y_pred_arima)
mse_arima = mean_squared_error(y_test, y_pred_arima)
rmse_arima = np.sqrt(mse_arima)

st.write("Evaluasi Model ARIMA:")
st.write(f"MAE: {mae_arima:.4f}")
st.write(f"MAPE: {mape_arima:.4f}")
st.write(f"MSE: {mse_arima:.4f}")
st.write(f"RMSE: {rmse_arima:.4f}")

# Memuat model LSTM
lstm_model = load_lstm_model()
scaler = MinMaxScaler()
data['Close_scaled'] = scaler.fit_transform(data[['Close']])

# Persiapan data untuk LSTM
train, test = data['Close_scaled'][:train_size], data['Close_scaled'][train_size:]
X_test, y_test = [], []
look_back = 1
for i in range(len(test) - look_back):
    X_test.append(test[i:(i + look_back)].values)
    y_test.append(test[i + look_back])
X_test = np.array(X_test).reshape(len(X_test), 1, look_back)
y_test = np.array(y_test)

# Prediksi dengan LSTM
y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# Visualisasi LSTM
st.subheader("Prediksi Harga Saham dengan LSTM")
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Harga Aktual')
plt.plot(test.index[look_back:], y_pred_lstm, label='Prediksi LSTM', color='green')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham')
plt.title('Prediksi Harga Saham dengan LSTM')
plt.legend()
st.pyplot(plt)

# Evaluasi LSTM
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
mape_lstm = mean_absolute_percentage_error(y_test, y_pred_lstm)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)

st.write("Evaluasi Model LSTM:")
st.write(f"MAE: {mae_lstm:.4f}")
st.write(f"MAPE: {mape_lstm:.4f}")
st.write(f"MSE: {mse_lstm:.4f}")
st.write(f"RMSE: {rmse_lstm:.4f}")
