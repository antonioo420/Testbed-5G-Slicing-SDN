import pandas as pd
from bitrate_utils import calculate_throughput
from dotenv import load_dotenv
import os

load_dotenv()
URL_STATS_1 = os.getenv('ODL_STATS1_URL')
URL_QUEUE_1 = os.getenv('ODL_QUEUE1_URL')
QUEUE_1 = os.getenv('QUEUE1_ID')

# Update parameters
INTERVAL = 5

import pandas as pd

def createDataset(iterations):    
    df = pd.DataFrame(columns=['throughput', 'timestamp'])
    time = 0

    try:
        for i in range(iterations):
            # Calcular throughput
            throughput = calculate_throughput(interval=INTERVAL, urlstats=URL_STATS_1)

            # Crear nueva fila
            nueva_fila = pd.DataFrame([[throughput, time]], columns=['throughput', 'timestamp'])

            print("Valores tomados: ", throughput, time)

            # Concatenar nueva fila
            df = pd.concat([df, nueva_fila.dropna(axis=1)], ignore_index=True)
            time += INTERVAL

    except KeyboardInterrupt:
        # Captura la interrupción de teclado (Ctrl+C)
        print("\nInterrupción detectada. Guardando los datos en 'dataset.csv'...")
        df.to_csv('dataset.csv', index=False)  # Guardar DataFrame en CSV
        print("Datos guardados exitosamente en 'dataset.csv'.")

    else:
        # Si no hubo interrupción, guarda el archivo después de que termine el bucle
        df.to_csv('dataset.csv', index=False)
        print("Datos guardados exitosamente en 'dataset.csv'.")

iterations = int(input("Introduce el número de valores que quieres tomar: "))
createDataset(iterations)