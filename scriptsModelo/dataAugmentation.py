import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.interpolate import CubicSpline
import random
import sys

def time_slicing_window(d, slice_start = 100, slice_end = 150):
    sliced_data = np.concatenate((d[0:slice_start], d[slice_end:-1]))
    return sliced_data

def add_gaussian_noise(time_series, mean=0.0, stddev=1.0):
    """
    Adds Gaussian noise to a time series.

     Options:
     time_series (array-like): A time series to which noise is added.
     mean (float): The average value of the noise. Default is 0.0.
     stddev (float): Standard deviation of noise. Default is 1.0.

     Returns:
     noisy_series (np.array): Time series with added noise.
     """
     # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))

    # Adding noise to the original time series
    noisy_series = time_series + noise

    return noisy_series

def add_scaling(time_series, scale_factor):
    """
    Scales a time series by multiplying each element by scale_factor.

    :param time_series: numpy array, time series to be scaled
    :param scale_factor: the number by which all elements of the series will be multiplied
    :return: numpy array, scaled time series
    """
    scaled_time_series = time_series * scale_factor
    return scaled_time_series

def magnitude_warping(time_series, num_knots=4, warp_std_dev=0.2):
    """
    Applies magnitude warping to a time series using cubic splines.

    :param time_series: np.array, time series to distort
    :param num_knots: int, number of control points for splines
    :param warp_std_dev: float, standard deviation for distorting the values of control points
    :return: np.array, distorted time series
    """
    # Generating random spline knots within a time series
    knot_positions = np.linspace(0, len(time_series) - 1, num=num_knots)
    knot_values = 1 + np.random.normal(0, warp_std_dev, num_knots)

    # Creating a Cubic Spline Function Through Knots
    spline = CubicSpline(knot_positions, knot_values)

    # Generating time indexes for a time series
    time_indexes = np.arange(len(time_series))

    # Applying distortion to a time series
    warped_time_series = time_series * spline(time_indexes)

    return warped_time_series

def time_warping(time_series, num_operations=10, warp_factor=0.2):
    """
    Applying time warping to a time series.

    :param time_series: Time series, numpy array.
    :param num_operations: Number of insert/delete operations.
    :param warp_factor: Warp factor that determines the impact of operations.
    :return: Distorted time series.
    """
    warped_series = time_series.copy()
    for _ in range(num_operations):
        operation_type = random.choice(["insert", "delete"])
        index = random.randint(1, len(warped_series) - 2)
        if operation_type == "insert":
            # Insert a value by interpolating between two adjacent points
            insertion_value = (warped_series[index - 1] + warped_series[index]) * 0.5
            warp_amount = insertion_value * warp_factor * random.uniform(-1, 1)
            warped_series = np.insert(warped_series, index, insertion_value + warp_amount)
        elif operation_type == "delete":
            # Remove a random point
            warped_series = np.delete(warped_series, index)
        else:
            raise ValueError("Invalid operation type")

    return warped_series

def shuffle_time_slices(time_series, slice_size=1):
    """
    Shuffle different time slices of the provided array.

    Parameters:
    time_series (array-like): An array containing time-series data.
    slice_size (int): The size of each time slice that will be shuffled.

    Returns:
    shuffled_data (array-like): The array with shuffled time slices.
    """

    # Convert to numpy array if not already
    time_series = np.array(time_series)

    if slice_size <= 0 or slice_size > len(time_series):
        raise ValueError("Slice size must be within the range 1 to len(data)")

    num_slices = len(time_series) // slice_size

    # Cut the data into slices
    slices = [time_series[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]

    # Shuffle the slices
    np.random.shuffle(slices)

    # Concatenate the shuffled slices
    shuffled_data = np.concatenate(slices)

    # Include any leftover data that couldn't form a complete slice
    remainder = len(time_series) % slice_size
    if remainder > 0:
        remainder_data = time_series[-remainder:]
        shuffled_data = np.concatenate([shuffled_data, remainder_data])

    return shuffled_data

def rotation(time_series, sigma=1.0):
    """
    Rotates a time series using an angle selected from normal distribution with standard deviation sigma.

    :param time_series: list or np.array, time series for augmentation
    :param sigma: float, standard deviation of the rotation angle in degrees
    :return: np.array, rotated time series
    """
    # Generating the rotation angle from the normal distribution
    angle = np.random.normal(scale=sigma)
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Creating a rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Let's transform the time series into a two-dimensional array, where each point is a pair (time, value)
    time_indices = np.arange(len(time_series)).reshape(-1, 1)
    values = np.array(time_series).reshape(-1, 1)
    two_dim_points = np.hstack((time_indices, values))

    # Apply the rotation matrix to each point in the time series
    rotated_points = two_dim_points.dot(rotation_matrix)

    # Returning only time series values after rotation
    return rotated_points[:, 1]

def load_and_augment(input_file, output_dir):
    # Load dataset
    dataset = pd.read_csv(input_file)
    data = dataset['throughput']

    d = np.array(data)
    
    augmented_data = []

    # Add noise
    #d_noisy = add_gaussian_noise(data, mean=0.0, stddev=0.05)    

    # Add scaling
    d_scaled = add_scaling(data, scale_factor=5)

    # Add magnitude warping
    #d_magnitude_warping = magnitude_warping(data, num_knots = 4, warp_std_dev=0.2)

    #Add time warping 
    #d_time_warping = time_warping(data, num_operations=10, warp_factor=0.2)

    # Shuffle data
    #d_shuffled = shuffle_time_slices(data, slice_size=20)

    #d_rotated = rotation(data, sigma=1)

    # Concatenar los datos aumentados
    augmented_data.append(d_scaled)
    augmented_data = np.concatenate(augmented_data, axis=0)

    # Guardar los datos aumentados como CSV
    augmented_df = pd.DataFrame(augmented_data, columns=["throughput"])
    augmented_df.to_csv(os.path.join(output_dir, f"augmented_{os.path.basename(input_file)}"), index=False)

    print(f"Archivo guardado en {output_dir}/augmented_{os.path.basename(input_file)}")

# Función principal para aplicar el augmentación a múltiples archivos
def process_datasets(input_files, output_dir):
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Aplicar augmentación a cada archivo CSV
    for input_file in input_files:
        print(f"Procesando archivo: {input_file}")
        load_and_augment(input_file, output_dir)

# Ejemplo de uso
if __name__ == "__main__":
    #input_files = ["dataset1000yt.csv"]  # Lista de archivos de entrada
    output_dir = "augmented_datasets"  # Carpeta donde se guardarán los nuevos CSV
    input_files = [sys.argv[1]]
    process_datasets(input_files, output_dir)
