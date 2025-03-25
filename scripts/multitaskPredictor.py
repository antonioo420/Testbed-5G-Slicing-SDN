import datetime
import requests
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, ReLU, SimpleRNN, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from collections import deque
import pandas as pd
from sklearn.metrics import mean_squared_error
import pdb
from plotDataset import plot_accuracy
import sys
import os
import subprocess, time, re

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
LOOKBACK = 30  # Number of previous data points to consider
ALPHA = 0.3  # EMA smoothing factor
N_CLASSES = 4 # youtube, twitch, prime, tiktok, navegacion web
#TRAFFIC_HISTORY = deque(maxlen=LOOKBACK)

# Define LSTM Model
def build_lstm_model():
    inputs = Input(shape=(LOOKBACK, 1))

    # Shared part
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(50)(x)        
    x = ReLU()(x) 
    #x = Dense(50)(x)        

    # Output 1: Throughput (regression)
    throughput_output = Dense(1, name='throughput_output')(x)

    # Output 2: Traffic type (classification)
    classification_output = Dense(N_CLASSES, activation='softmax', name='classification_output')(x)

    # Final model
    model = Model(inputs=inputs, outputs=[throughput_output, classification_output])

    return model

def build_rnn_model():
    inputs = Input(shape=(LOOKBACK, 1))

    # Shared part
    x = SimpleRNN(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = SimpleRNN(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = SimpleRNN(50, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(50)(x)        
    x = ReLU()(x) 
    #x = Dense(50)(x)        

    # Output 1: Throughput (regression)
    throughput_output = Dense(1, name='throughput_output')(x)

    # Output 2: Traffic type (classification)
    classification_output = Dense(N_CLASSES, activation='softmax', name='classification_output')(x)

    # Final model
    model = Model(inputs=inputs, outputs=[throughput_output, classification_output])

    return model

def build_gru_model():
    inputs = Input(shape=(LOOKBACK, 1))

    # Shared part
    x = GRU(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = GRU(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = GRU(50, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(50)(x)        
    x = ReLU()(x) 
    #x = Dense(50)(x)        

    # Output 1: Throughput (regression)
    throughput_output = Dense(1, name='throughput_output')(x)

    # Output 2: Traffic type (classification)
    classification_output = Dense(N_CLASSES, activation='softmax', name='classification_output')(x)

    # Final model
    model = Model(inputs=inputs, outputs=[throughput_output, classification_output])

    return model

def normalise_dataset(X, max, min):
    # MinMaxScaler
    X_std = (X - min) / (max - min)
    #Replace 1 and 0 with max and min values
    X_scaled = X_std * (1 - 0) + 0
   
    return X_scaled

def load_split_dataset(input_file, norm = 1):
    x_train = []
    y_train_throughput = []
    y_train_class = []
    x_test = []
    y_test_throughput = []
    y_test_class = []

    dataset = pd.read_csv(input_file)
    throughput = dataset['throughput']  
    class_ = dataset['class']      

    if(norm == 1):
        max = np.max(throughput)
        min = np.min(throughput)
        throughput = normalise_dataset(throughput, max, min)
    
    # Calc index that splits the 80% of data
    split_index = int(len(throughput) * 0.8)

    # 80% data for training, 20% for testing
    throughput_train = throughput[:split_index]
    throughput_test = throughput[split_index:]  

    class_train = class_[:split_index]
    class_test = class_[split_index:]                

    # Create sequences for training set
    for i in range(LOOKBACK, len(throughput_train)):        
        x_train.append(throughput_train[i-LOOKBACK:i])  
        y_train_throughput.append(throughput_train[i])            
        y_train_class.append(class_train[i])  
    
    # Create sequences for testing set
    for j in range(LOOKBACK, len(throughput_test)):                               
        x_test.append(throughput_test.iloc[j-LOOKBACK:j]) 
        y_test_throughput.append(throughput_test.iloc[j])  
        y_test_class.append(class_test.iloc[j])       
    
    # Convert to numpy array
    x_train = np.array(x_train)
    y_train_throughput = np.array(y_train_throughput)
    y_train_class = np.array(y_train_class)
    x_test = np.array(x_test)
    y_test_throughput = np.array(y_test_throughput)    
    y_test_class = np.array(y_test_class)

    # Reshape so the LSTM input has the shape (n_samples, time_steps, n_features)    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))    

    return x_train, y_train_throughput, y_train_class, x_test, y_test_throughput, y_test_class

def load_dataset(input_file, norm = 1):
    x = []
    y_throughput = []
    y_class = []

    dataset = pd.read_csv(input_file)
    throughput = dataset['throughput']
    class_ = dataset['class']

    if norm == 1:
        max = np.max(throughput)
        min = np.min(throughput)
        throughput = normalise_dataset(throughput, max, min)

    for i in range(LOOKBACK, len(throughput)):        
        x.append(throughput.iloc[i-LOOKBACK:i])  
        y_throughput.append(throughput.iloc[i])            
        y_class.append(class_.iloc[i])            

    x = np.array(x)
    y_throughput = np.array(y_throughput)
    y_class = np.array(y_class)

    x = x.reshape((x.shape[0], x.shape[1], 1))

    return x, y_throughput, y_class

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=5):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

def train_model(x_train, y_train_throughput, y_train_class, model, epochs = 100, batch_size = 32, path = './traffic_predictor_nodropout.h5'):
    optimizer = Adam()

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss={
            'throughput_output': 'mse',  # Mean Squared Error for regression
            'classification_output': 'sparse_categorical_crossentropy',  # Cross entropy for classification (-log[p(y)])
        },
        metrics={
            'throughput_output': ['mae'],  # Mean Absolute Error as metric for regression
            'classification_output': ['accuracy'],  # Accuracy as metric for classification
        }
    )
    #model.summary()
    lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=5)

    model.fit(
        x_train,
        {'throughput_output': y_train_throughput, 'classification_output': y_train_class},
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1, 
        callbacks=[lr_sched]
    )
    model.save(path)
    print("Model trained and saved.")

def accuracy(x_test, y_test_throughput, y_test_class, model, path='test.png'):  
    y_pred_throughput, y_pred_class = model.predict(x_test)
        
    plot_accuracy(y_pred_throughput, y_test_throughput, path)
    
    rmse = np.sqrt(mean_squared_error(y_test_throughput, y_pred_throughput))
    
    y_pred_class_labels = np.argmax(y_pred_class, axis=1)
    
    acc = accuracy_score(y_test_class, y_pred_class_labels)
    
    print(f"RMSE (Throughput): {rmse:.4f}")
    print(f"Accuracy (Classification): {acc:.4f}")

    #return rmse, acc

def getclass(class_value):
    match class_value:
        case 0:
            return 'youtube'
        case 1:
            return 'twitch'
        case 2:
            return 'prime'
        case 3:
            return 'tiktok'
        case 4:
            return 'navegacion web'

# Hasta 300 ms de frecuencia de actualización
def get_nload_throughput(interface, model):
    # Ejecuta nload en modo no interactivo con actualización cada 500ms y utiliza Mb/s como unidad
    # nload -u m -m -t 500 <interface>
    command = ["nload", "-u", "m", "-m", "-t", "500", interface]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    window = deque(maxlen=LOOKBACK)
    try:
        for line in process.stdout:
            if not line:
                break

            # Buscar valores de Curr:
            incoming_match = re.search(r"Curr:\s+([\d.]+) (\w+)", line)            
            if incoming_match:                
                incoming_value = incoming_match.group(1)
                print(incoming_value)

                window.append(float(incoming_value))

                if len(window) == LOOKBACK:
                    input_data = np.array(window).reshape(1, LOOKBACK, 1)
                    throughput_pred, class_pred = model.predict(input_data)

                    throughput = throughput_pred[0]  # Predicción del throughput
                    class_value = np.argmax(class_pred[0])  # Clase con mayor probabilidad

                    class_ = getclass(class_value)
                    print(f"Predicción throughput: {throughput[0]:.2f}")
                    print(f"Predicción clase: {class_}")
                    #max_rate = int(prediction * 1.2)  # Buffer factor
                    #update_queue_max_rate(max_rate)

    except KeyboardInterrupt:
        # Captura la interrupción de teclado (Ctrl+C)
        print(f"\nInterrupción detectada")        
        process.terminate()

if __name__ == "__main__":
    try:
        model = load_model("./zprueba_look30_4clases/traffic_predictor.h5", custom_objects={'mse': MeanSquaredError()})
        print("----------Modelo cargado--------------")
    except:
        model = build_lstm_model()
        print("----------Modelo creado--------------")
    #print(model.summary())
    try:
        dataset = sys.argv[1]
        option = sys.argv[2]

        match option:
            case '0':
                x_train, y_train_throughput, y_train_class, x_test, y_test_throughput, y_test_class = load_split_dataset(dataset, norm = 1)
                train_model(x_train, y_train_throughput, y_train_class, model, epochs=25, batch_size=32)
                accuracy(x_test, y_test_throughput, y_test_class, model)
            case '1':
                x_train, y_train_throughput, y_train_class = load_dataset(dataset, norm = 0)
                train_model(x_train, y_train_throughput, y_train_class, model, epochs=25, batch_size=32)
            case '2':
                x_test, y_test_throughput, y_test_class = load_dataset(dataset, norm = 0)
                accuracy(x_test, y_test_throughput, y_test_class, model)
    except IndexError:
        get_nload_throughput('eno8303', model)
