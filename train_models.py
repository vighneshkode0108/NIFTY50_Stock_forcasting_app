import os
import warnings
import numpy as np
import pandas as pd
import configparser
warnings.filterwarnings("ignore")
from utils import nifty50_companies
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

config = configparser.ConfigParser()
config.read("config.ini")

no_of_epochs = int(config['model']['no_of_epochs'])
batch_size = int(config['model']['batch_size'])

def preprocess_data(data):
    # Ensure 'Close' column exists and is numeric
    if 'Close' not in data.columns:
        raise ValueError("Missing 'Close' column in data.")
    
    # Convert 'Close' column to numeric (will convert invalid strings to NaN)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # Drop any rows with NaN values in 'Close'
    data = data.dropna(subset=['Close'])

    # Check again if data is now empty
    if data.empty:
        raise ValueError("No valid numeric data in 'Close' column after cleaning.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def build_lstm_model(X, Y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size = batch_size, epochs = no_of_epochs)

    return model

def train_and_save_models():
    for company, ticker in nifty50_companies.items():
        filename = f"data/{ticker}.csv"
        model_filename = f"models/{ticker}_model.keras"
        
        # Check if data file exists
        if os.path.exists(filename):
            # Load data
            print(f"Training started for {company}.. Please wait..!!")
            # data = pd.read_csv(filename, index_col='Date', parse_dates=True)
            df = pd.read_csv(filename)
            if 'Date' not in df.columns:
              print(f"Skipping {company} — 'Date' column missing.")
              continue

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            data = df

            scaled_data, scaler = preprocess_data(data)
            
            # Prepare dataset
            X, Y = [], []
            for i in range(len(scaled_data) - 100 - 1):
                X.append(scaled_data[i:(i + 100), 0])
                Y.append(scaled_data[i + 100, 0])
            X, Y = np.array(X), np.array(Y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build and train model
            model = build_lstm_model(X, Y)
            
            # Save model
            model.save(model_filename)
            print(f"Saved model for {company} as {model_filename}.")
            print("-----------------------------------------------")
        else:
            print(f"No data file found for {company}.")

def main():
    train_and_save_models()

if __name__ == "__main__":
    main()
    