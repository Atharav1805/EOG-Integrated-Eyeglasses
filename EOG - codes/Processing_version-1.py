import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter

# Function to design a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply bandpass filter to data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# File to read data from
blink_data = 'eog_data_blink.csv'

# Initialize lists to store data
timestamps = []
voltages = []

# Read data from the CSV file for plotting
with open(blink_data, mode='r') as file: # Open the file for reading
    reader = csv.reader(file) # Create a CSV reader object
    next(reader)  # Skip the header row
    for row in reader:
        timestamps.append(float(row[0]))
        voltages.append(float(row[1]))

# Convert lists to numpy arrays for processing
timestamps = np.array(timestamps)
voltages = np.array(voltages)

# Sampling rate (assuming timestamps are in milliseconds)
sampling_rate = 1000 / (timestamps[1] - timestamps[0])

# Perform Fourier Transform
N = len(voltages)
T = 1.0 / sampling_rate
yf = fft(voltages)
xf = fftfreq(N, T)[:N//2]

# Plot the time-domain signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(timestamps, voltages)
plt.title('Time Domain Signal')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage (V)')

# Plot the frequency spectrum
plt.subplot(3, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid()

# Identify the dominant frequencies
dominant_freqs = xf[np.argsort(-np.abs(yf[:N//2]))][:5]
print("Dominant frequencies:", dominant_freqs)

# Apply bandpass filter to remove noise
lowcut = 0.1  # Low cut-off frequency in Hz
highcut = 30.0  # High cut-off frequency in Hz
filtered_signal = bandpass_filter(voltages, lowcut, highcut, sampling_rate, order=6)

# Perform inverse Fourier Transform to get the filtered signal back to time domain
filtered_signal_time_domain = ifft(yf).real

# Plot the filtered time-domain signal
plt.subplot(3, 1, 3)
plt.plot(timestamps, filtered_signal)
plt.title('Filtered Time Domain Signal')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage (V)')
plt.tight_layout()
plt.show()

# Blink detection based on thresholding
threshold = np.mean(filtered_signal) + 2 * np.std(filtered_signal)
blink_indices = np.where(filtered_signal > threshold)[0]
blink_times = timestamps[blink_indices]

# Print and save blink times
print("Blink times:", blink_times)

# Save blink times to a file
with open('blink_times.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Blink Time (ms)'])
    for blink_time in blink_times:
        writer.writerow([blink_time])

# Save filtered signal to a file
with open('filtered_signal.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (ms)', 'Filtered Voltage (V)'])
    for time, voltage in zip(timestamps, filtered_signal):
        writer.writerow([time, voltage])