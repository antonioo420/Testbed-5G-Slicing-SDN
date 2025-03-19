import datetime
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import pandas as pd
from sklearn.metrics import mean_squared_error
import pdb
from plotDataset import plot_accuracy
import sys
import os
import yfinance as yf

# OpenDaylight RESTCONF Credentials
CONTROLLER_IP = "<controller-ip>"
USERNAME = "admin"
PASSWORD = "admin"

# Queue details
QUEUE_UUID = "dafbcf5f-0b94-491f-9e17-7f30007c370a"
HOSTNAME = "HOST1"

# API Endpoints
BASE_URL = f"http://{CONTROLLER_IP}:8181/restconf/config/network-topology:network-topology/topology/ovsdb:1/node/ovsdb%3A%2F%2F{HOSTNAME}/ovsdb:queues/queue%3A%2F%2F{QUEUE_UUID}/"

# Parameters
LOOKBACK = 10  # Number of previous data points to consider
ALPHA = 0.3  # EMA smoothing factor
#TRAFFIC_HISTORY = deque(maxlen=LOOKBACK)

# Define LSTM Model
def build_lstm_model():
    # model = Sequential([
    #     LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    #     LSTM(50, return_sequences=False),
    #     #Dense(25, activation='relu'),
    #     Dense(1)
    # ])
    # model = Sequential([
    #     LSTM(256, return_sequences=True, input_shape=(LOOKBACK, 1)),
    #     Dropout(0.2),
    #     LSTM(128, return_sequences=True),
    #     Dropout(0.2),
    #     LSTM(64, return_sequences=False),
    #     Dense(1)
    # ])    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.5),
        LSTM(50, return_sequences=False),
        Dropout(0.5),
        Dense(1)
    ])
    return model

def get_max_min_datasets(dir):
    dataset_total = []

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            # print(os.path.join(directory, filename))
            dataset = pd.read_csv(dir + '/' + filename)
            throughput = dataset['throughput']   
            dataset_total.append(throughput)

    max = np.max(throughput)
    min = np.min(throughput)

    return max, min

def normalise_dataset(X, max, min):
    # MinMaxScaler
    X_std = (X - min) / (max - min)
    #Replace 1 and 0 with max and min values
    X_scaled = X_std * (1 - 0) + 0

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # # Reshape for scale
    # input_normalised = input.values.reshape(-1, 1)
    # # Apply scale
    # input_normalised = scaler.fit_transform(input_normalised)
    #     
    return X_scaled

def load_split_dataset(input_file, norm = 1):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    dataset = pd.read_csv(input_file)
    throughput = dataset['throughput']        

    if(norm == 1):
        max, min = get_max_min_datasets(os.path.dirname(input_file))
        throughput = normalise_dataset(throughput, max, min)
    
    # Calc index that splits the 80% of data
    split_index = int(len(throughput) * 0.8)

    # 80% data for training, 20% for testing
    throughput_train = throughput[:split_index]
    throughput_test = throughput[split_index:]                

    # Create sequences for training set
    for i in range(LOOKBACK, len(throughput_train)):        
        x_train.append(throughput_train[i-LOOKBACK:i])  
        y_train.append(throughput_train[i])            
    
    # Create sequences for testing set
    for j in range(LOOKBACK, len(throughput_test)):                               
        x_test.append(throughput_test.iloc[j-LOOKBACK:j]) 
        y_test.append(throughput_test.iloc[j])        
    
    # Convert to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)    

    # Reshape so the LSTM input has the shape (n_samples, time_steps, n_features)
    # Add an extra dimension for characteristics (in this case, only 'throughput')
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    

    return x_train, y_train, x_test, y_test

def load_dataset(input_file, norm = 1):
    x = []
    y = []

    dataset = pd.read_csv(input_file)
    throughput = dataset['throughput']
    #throughput = throughput[1000:1400]
    if norm == 1:
        max, min = get_max_min_datasets(os.path.dirname(input_file))
        throughput = normalise_dataset(throughput, max, min)

    for i in range(LOOKBACK, len(throughput)):        
        x.append(throughput.iloc[i-LOOKBACK:i])  
        y.append(throughput.iloc[i])            

    x = np.array(x)
    y = np.array(y)

    x = x.reshape((x.shape[0], x.shape[1], 1))

    return x, y

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=5):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

def train_model(x_train, y_train, model, epochs = 100, batch_size = 32, path = './traffic_predictor.h5'):
    """Train LSTM model on simulated traffic data."""
    optimizer = Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse')

    lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=5)

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[lr_sched])
    model.save(path)
    print("Model trained and saved.")

def accuracy(x_test, y_test, model, path = 'test.png'):
    #Predict with whole test dataset    
    y_pred = model.predict(x_test)    
    plot_accuracy(y_pred, y_test, path)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))    
    return rmse

# def predict_real_time(x):            
#     input_data = np.array(x.reshape(1, LOOKBACK))
#     prediction = model.predict(input_data)[0][0]
#     return prediction
    
if __name__ == "__main__":
    dataset = sys.argv[1] 
    option = sys.argv[2]
    print(os.path.dirname(dataset))
    print(option)
        # Load pre-trained model or train from scratch
    try:    
        model = load_model("./traffic_predictor.h5", custom_objects={'mse': MeanSquaredError()})
        print("----------Modelo cargado--------------")
    except:
        model = build_lstm_model()    
        print("----------Modelo creado--------------")

    match option:
        case '0':
            x_train, y_train, x_test, y_test = load_split_dataset(dataset, norm = 1)   
            train_model(x_train, y_train, model, epochs=25, batch_size=32)     
            rmse = accuracy(x_test, y_test, model)
            print(rmse)
        case '1':
            x_train, y_train = load_dataset(dataset, norm = 1)
            train_model(x_train, y_train, model, epochs=25, batch_size=32)
        case '2':
            x_test, y_test = load_dataset(dataset, norm = 1)
            rmse = accuracy(x_test, y_test, model)
            print(rmse)