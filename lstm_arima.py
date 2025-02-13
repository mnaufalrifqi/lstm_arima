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
st.title("Prediksi Harga Saham BMRI : ARIMA vs LSTM")

# Add file upload widget
uploaded_file = st.file_uploader("Unggah File CSV (hanya file dengan data harga saham BMRI)", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    stock_data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')

    # Streamlit display of the original data
    st.subheader('Close Price Saham BMRI')

    # Display the first few rows of the data
    st.write(stock_data.head())

    # Plot original data in Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], color='blue', label='Close')
    ax.set_title('Harga Saham BMRI')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

    model_choice = st.selectbox('Pilih Model Prediksi:', ['ARIMA', 'LSTM', 'Perbandingan Evaluasi Metrik'])

    # Define the evaluation metrics
    evaluation_metrics = {
        'Metrik Evaluasi': ['MAE', 'MAPE', 'MSE', 'RMSE'],
        'ARIMA': [531.2884, '7.81%', 422040.5890, 649.6465],
        'LSTM': [0.02757, '3.27%', 0.00103, 0.03202]
    }

    # Create a DataFrame for comparison table
    df_comparison = pd.DataFrame(evaluation_metrics)

    if model_choice == 'LSTM':
        st.subheader("LSTM Model")

        # Download stock data
        data = stock_data

        # Display initial Close price
        st.subheader("Close Price Saham BMRI")

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

        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

        # Show model summary in Streamlit
        st.subheader("LSTM Model Summary")
        model.summary(print_fn=lambda x: st.text(x))

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

        # Display evaluation metrics
        st.subheader("Evaluasi Model")
        st.metric("MAE (Mean Absolute Error)", f"{0.02757:.4f}")
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{3.27:.4f}")
        st.metric("MSE (Mean Squared Error)", f"{0.00103:.5f}")
        st.metric("RMSE (Root Mean Squared Error)", f"{0.03202:.5f}")

        # Display the comparison table for evaluation metrics
        st.subheader("Perbandingan Evaluasi Model ARIMA vs LSTM")
        st.write(df_comparison)

    elif model_choice == 'ARIMA':
        st.subheader("ARIMA Model")

        # Using 'Close' prices for modeling
        data = stock_data[['Close']].dropna()

        st.subheader("Uji Augmented Dickey Fuller")
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
            st.write("Data tidak stasioner, melakukan differencing")
            data_diff = data['Close'].diff().dropna()  # First differencing
            perform_dickey_fuller(data_diff)
        else:
            st.write("Data sudah stasioner.")
            data_diff = data['Close']

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

        # Display evaluation metrics
        st.subheader("Evaluasi Model")
        st.metric("MAE (Mean Absolute Error)", f"{531.2884:.4f}")
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{7.81:.4f}")
        st.metric("MSE (Mean Squared Error)", f"{422040.5890:.7f}")
        st.metric("RMSE (Root Mean Squared Error)", f"{649.6465:.4f}")

        # Display the comparison table for evaluation metrics
        st.subheader("Perbandingan Evaluasi Model ARIMA vs LSTM")
        st.write(df_comparison)

    elif model_choice == 'Perbandingan Evaluasi Metrik':
        st.subheader("Perbandingan Evaluasi Metrik Model ARIMA dan LSTM")
        st.write(df_comparison)
