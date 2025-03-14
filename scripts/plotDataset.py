import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_csv_data(file_path):    
    # Load CSV file in a DataFrame
    df = pd.read_csv(file_path)

    # Check throughput and timestamp columns exist
    if 'throughput' not in df.columns:
        print("El CSV no tiene las columnas esperadas.")
        return

    # Extract columns
    throughput = df['throughput']
    #throughput = throughput[1000:1200]

    # Create plot
    plt.figure(figsize=(200, 12))
    plt.plot(throughput, color='b', label='Throughput')

    # Set labels and title
    plt.xlabel('Tiempo')
    plt.title('Gráfico de Throughput')
    plt.legend()
    
    # Show graphic
    plt.grid(True)

    file_path = file_path[:len(file_path)-4] # Obtain file_path name without '.csv'    
    plt.savefig(file_path+'.png')

def plot_accuracy(y_pred, y_test, path):            
    plt.figure(figsize=(300, 12))
    plt.plot(y_test, '-o', label='Valores reales')
    plt.plot(y_pred, '-o', label='Predicciones')
    
    plt.ylabel('Throughput (Mbps)')
    plt.title('Valores reales vs predicciones')
    plt.legend()
        
    plt.grid(True)
    plt.savefig(path)

if __name__ == "__main__":
    file_path = sys.argv[1]
    plot_csv_data(file_path)