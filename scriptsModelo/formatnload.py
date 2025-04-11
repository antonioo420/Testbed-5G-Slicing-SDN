import re
import sys

if __name__ == "__main__":
    # Archivo de entrada y salida
    input_file = sys.argv[1]
    output_file = "throughput.csv"
    # Expresión regular para extraer los valores de "Curr:"
    pattern = re.compile(r'Curr:\s+([\d\.]+)\s+MBit/s')

    # Lista para almacenar los valores de incoming
    incoming_values = []

    # Leer el archivo y extraer los valores
    with open(input_file, "r") as file:
        for line in file:
            matches = pattern.findall(line)            
            if len(matches) >= 2:  # Se asume que el primer valor es "incoming" y el segundo es "outcoming"
                incoming_values.append(matches[0])  # Guardamos solo el primero (incoming)

    # Escribir en el archivo CSV
    with open(output_file, "w") as file:
        file.write("throughput\n")  # Encabezado
        file.write("\n".join(incoming_values))  # Valores

    print(f"Archivo '{output_file}' generado con {len(incoming_values)} valores.")
