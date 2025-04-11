from scriptPythonPaper import build_lstm_model, load_dataset, load_split_dataset, train_model, accuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
if __name__ == "__main__":
    # try:    
    #     model = load_model("./trained_4000.h5", custom_objects={'mse': MeanSquaredError()})
    #     print("----------Modelo cargado--------------")
    # except:
    #     model = build_lstm_model()    
    #     print("----------Modelo creado--------------")

    # match option:
    #     case '0':
    #         x_train, y_train, x_test, y_test = load_split_dataset(dataset, norm = 1)   
    #         train_model(x_train, y_train, model, epochs=25, batch_size=32)     
    #         rmse = accuracy(x_test, y_test)
    #         print(rmse)
    #     case '1':
    #         x_train, y_train = load_dataset(dataset, norm = 1)
    #         train_model(x_train, y_train, model, epochs=25, batch_size=32)
    #     case '2':
    #         x_test, y_test = load_dataset(dataset, norm = 1)
    #         rmse = accuracy(x_test, y_test)
    #         print(rmse)

    historico_rmse = []
    historico_rmse.append('------------Historico rmse------------')

    try:
        os.makedirs('resultados/iperf/tcp')
        os.makedirs('resultados/iperf/udp')
        os.makedirs('resultados/nload/youtube')
        os.makedirs('resultados/nload/twitch')
    except:
        print("Directorios creados")

    ################### iperf - tcp #############################
    historico_rmse.append('------------iperf------------')
    historico_rmse.append('------------tcp------------')

    ruta_data_tcp = 'datasets/iperf/tcp/'
    ruta_model_tcp = 'modelos/iperf/tcp/tcp.h5'
    print(os.path.exists(ruta_model_tcp))
    if not os.path.exists(ruta_model_tcp):
        x_train, y_train = load_dataset(ruta_data_tcp + 'dataset400_tcp.csv', norm = 1)
        model = build_lstm_model()    
        train_model(x_train, y_train, model, epochs=25, batch_size=32, path = ruta_model_tcp)
    
    model = load_model(ruta_model_tcp, custom_objects={'mse': MeanSquaredError()})
    x_test, y_test = load_dataset(ruta_data_tcp + 'dataset300_tcp.csv', norm = 1)
    rmse = accuracy(x_test, y_test, model, path = 'resultados/iperf/tcp/test.png') 
    historico_rmse.append(rmse)

    ################### iperf - udp #############################
    historico_rmse.append('------------udp------------')

    ruta_data_udp = 'datasets/iperf/udp/'
    ruta_model_udp = 'modelos/iperf/udp/udp.h5'

    if not os.path.exists(ruta_model_udp):
        x_train, y_train = load_dataset(ruta_data_udp + 'dataset300_udp.csv', norm = 1)
        model = build_lstm_model()    
        train_model(x_train, y_train, model, epochs=25, batch_size=32)
 
    model = load_model(ruta_model_udp, custom_objects={'mse': MeanSquaredError()})
    x_test, y_test = load_dataset(ruta_data_udp + 'dataset200_udp.csv', norm = 1)
    rmse = accuracy(x_test, y_test, model, path = 'resultados/iperf/udp/test.png') 
    historico_rmse.append(rmse)


    # Escribir en el archivo CSV
    with open('rmse.txt', "w") as file:
        file.write("\n".join(historico_rmse)) 