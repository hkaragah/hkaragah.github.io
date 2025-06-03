###############################
# Earthquake Data Processing Module
# Written by: Hossein Karagah
# Date: 2025-05-25
# Description: This module provides functions to read, process, and analyze earthquake data.
###############################


import numpy as np
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def read_values(file_path, dt, first_line):
    """
    Read the data file starting from the first_line.
    
    Args:
        file_path (str): Path to the data file
        dt (float): Time step between data points
        first_line (int): Line number where data starts (1-based indexing)
    
    Returns:
        tuple: (time_array, data_array, header)
            - time_array: Array of time points
            - data_array: Array of values
            - header: List of header lines
    """
    with open(file_path, 'r') as file:
        # Read header
        header = []
        for _ in range(first_line - 1):
            header.append(file.readline().strip())
        
        # Read data values
        values = []
        for line in file:
            line_values = [float(val) for val in line.strip().split()]
            values.extend(line_values)
        
        # Convert to numpy arrays
        data = np.array(values)
        time = np.arange(0, len(data) * dt, dt)
    
    return time, data, header


def read_time_values(file_path, first_line):
    """
    Read a file containing time-value pairs starting from the specified line.
    
    Args:
        file_path (str): Path to the data file
        first_line (int): Line number where data starts (1-based indexing)
    
    Returns:
        tuple: (time_array, data_array, header)
            - time_array: Array of time points
            - data_array: Array of values
            - header: List of header lines
            
    Raises:
        ValueError: If any line doesn't contain exactly two values
    """
    with open(file_path, 'r') as file:
        # Read header
        header = []
        for _ in range(first_line - 1):
            header.append(file.readline().strip())
        
        # Read time-value pairs
        time_values = []
        for line in file:
            line_values = [float(val) for val in line.strip().split()]
            if len(line_values) != 2:
                raise ValueError(f"Expected 2 values per line (time and data), but got {len(line_values)} values")
            time_values.append(line_values)
        
        # Convert to numpy arrays efficiently
        time_values_array = np.array(time_values)
        time = time_values_array[:, 0]  # First column is time
        data = time_values_array[:, 1]  # Second column is data
    
    return time, data, header


def read_peer_nga_file(file_path):
    """
    Read specifically formatted PEER NGA acceleration data file.
    
    Args:
        file_path (str): Path to the PEER NGA file
        
    Returns:
        tuple: (time_array, acceleration_array, metadata)
            - time_array: Array of time points
            - acceleration_array: Array of acceleration values
            - metadata: Dictionary containing file information
    """
    metadata = {}
    
    # Read the file using the general reader
    time, acc, header = read_values(file_path, dt=0.005, first_line=5)
    
    # Parse metadata from header
    metadata['title'] = header[0]
    metadata['event_info'] = header[1]
    metadata['data_type'] = header[2]
    
    # Parse NPTS and DT from the fourth line
    header_info = header[3].split(',')
    metadata['npts'] = int(header_info[0].split('=')[1])
    metadata['dt'] = float(header_info[1].split('=')[1].split()[0])
    metadata['data_start_line'] = 5
    metadata['total_points'] = len(acc)
    metadata['duration'] = time[-1]
    
    return time, acc, metadata


def integrate_acceleration(acc, dt, dt_refiner_factor=1, baseline_correction=False, highpass_filter=False):
    """
    Integrate acceleration time history to get instantaneous velocity and displacement
    
    Parameters:
    acc (array): Acceleration time history
    dt (float): Time step
    
    Returns:
    vel (array): Instantaneous velocity at each time step
    disp (array): Instantaneous displacement at each time step
    """

    if highpass_filter:
        fs = 1 / dt # Sampling frequency
        acc = apply_highpass_filter(acc, fs, cutoff=0.05)
        
    if baseline_correction:
        acc= apply_baseline_correction(acc, dt, method='polyfit')
        
    # Initialize velocity and displacement arrays
    vel = np.zeros_like(acc)
    disp = np.zeros_like(acc)
        
    # Create finer time array with reduced time interval
    t_orig = np.arange(0, len(acc)*dt, dt)
    dt_fine = dt/dt_refiner_factor  # Reduce time step by dt_refiner_factor
    t_fine = np.arange(0, t_orig[-1], dt_fine)  # Remove endpoint to match array lengths
    
    # Interpolate acceleration to finer time grid
    acc_interp = interp1d(t_orig, acc, kind='cubic', bounds_error=False, fill_value='extrapolate')
    acc_fine = acc_interp(t_fine)
    
    # Integrate interpolated acceleration to get velocity
    vel_fine = cumulative_trapezoid(acc_fine, dx=dt_fine, initial=0)
    
    # Interpolate back to original time grid
    vel = interp1d(t_fine, vel_fine, bounds_error=False, fill_value='extrapolate')(t_orig)
    
    # Integrate velocity to get displacement
    disp_fine = cumulative_trapezoid(vel_fine, dx=dt_fine, initial=0)
    disp = interp1d(t_fine, disp_fine, bounds_error=False, fill_value='extrapolate')(t_orig)        
        
    return vel, disp
    

def apply_baseline_correction(acc, dt, method='polyfit'):
    if method == 'polyfit':
        # Fit a polynomial to the acceleration data
        t_orig = np.arange(0, len(acc) * dt, dt)
        coeffs = np.polyfit(t_orig, acc, deg=2)
        acc -= np.polyval(coeffs, t_orig)
    elif method == 'mean':
        # Subtract the mean value
        acc -= np.mean(acc)
    return acc


def apply_highpass_filter(data, fs, cutoff=0.05):
    nyquist = fs / 2
    b, a = butter(4, cutoff / nyquist, btype='high', analog=False)
    return filtfilt(b, a, data)


def main():
    pass
    
    
if __name__ == "__main__":
    main()