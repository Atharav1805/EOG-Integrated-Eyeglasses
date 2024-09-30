import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter, find_peaks
import neurokit2 as nk



# Function to design a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply bandpass filter to data
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data) 
    return y

# File to read data from
eog_signal = 'eog_data_sides_2.csv'

# Initialize lists to store data
timestamps = []
voltages = []

# Read data from the CSV file for plotting
with open(eog_signal, mode='r') as file: # Open the file for reading
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

# Apply bandpass filter to remove noise
lowcut = 0.2  # Low cut-off frequency in Hz
highcut = 5.0  # High cut-off frequency in Hz
filtered_signal = bandpass_filter(voltages, lowcut, highcut, sampling_rate, order=4)

# Compute the Fourier Transform of the filtered signal
N = len(filtered_signal)
T = 1.0 / sampling_rate
yf = fft(filtered_signal)
xf = fftfreq(N, T)[:N//2]

# Plot the original and filtered time-domain signal
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(timestamps, voltages)
plt.title('Original Time Domain Signal')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage (V)')

plt.subplot(3, 1, 2)
plt.plot(timestamps, filtered_signal)
plt.title('Filtered Time Domain Signal')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage (V)')

# Plot the frequency spectrum of the filtered signal
plt.subplot(3, 1, 3)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('Frequency Spectrum of Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.show()

# Amplitude thresholding to keep signals of certain amplitude
amplitude_threshold = np.mean(filtered_signal) + 2 * np.std(filtered_signal)
thresholded_signal = np.where(np.abs(filtered_signal) > amplitude_threshold, filtered_signal, 0)

# Blink detection using peak finding
peak_indices, _ = find_peaks(thresholded_signal, height=amplitude_threshold)
blink_times_pf = timestamps[peak_indices] # Blink times using method 1


# Blink detection based on thresholding
threshold = np.mean(filtered_signal) + 2 * np.std(filtered_signal)
blink_indices_thresh = np.where(filtered_signal > threshold)[0] # Blink indices using method 2
blink_times_thresh = timestamps[blink_indices_thresh] # Blink times using method 2

#---------------------------------------------------------------------------------------------------------------------------------
# Print and save blink times
print(f"Blink timestamps: \n1. Using Peak finding: {blink_times_pf} \n2. Using Thresholding: {blink_times_thresh}", end = ' ')

# Save blink times to a file
with open('blink_times_using_peak_finding.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Blink Time (ms) - Peak Finding'])
    for blink_time in blink_times_pf:
        writer.writerow([blink_times_pf])

with open('blink_times_using_thresholding.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Blink Time (ms) - Thresholding'])
    for blink_time in blink_times_thresh:
        writer.writerow([blink_times_thresh])
#---------------------------------------------------------------------------------------------------------------------------------


# Save filtered signal to a file
with open('filtered_signal.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (ms)', 'Filtered Voltage (V)'])
    for time, voltage in zip(timestamps, filtered_signal):
        writer.writerow([time, voltage])



# Plot the thresholded signal with detected blinks
plt.figure(figsize=(12, 6))
plt.plot(timestamps, thresholded_signal, label='Thresholded Signal')
plt.plot(timestamps[peak_indices], thresholded_signal[peak_indices], 'rx', label='Detected Blinks')
plt.title('Thresholded Signal(Amplitude Threshholding) with Detected Blinks')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()


included_cols = [1]
with open(eog_signal, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        content = [row[i] for i in included_cols]
        


print(content)
nk.signal_plot(content)
plt.show()
eog_cleaned = nk.eog_clean(content, sampling_rate=100, method='neurokit')
nk.signal_plot([content[100:1700], eog_cleaned[100:1700]], labels=["Raw Signal", "Cleaned Signal"])
plt.show()
blinks = nk.eog_findpeaks(eog_cleaned, sampling_rate=100, method="mne")

events = nk.epochs_create(eog_cleaned, blinks, sampling_rate=100, epochs_start=-0.3, epochs_end=0.7)

data = nk.epochs_to_array(events)  # Convert to 2D array
data = nk.standardize(data)  # Rescale so that all the blinks are on the same scale

# Plot with their median (used here as a robust average)  
plt.plot(data, linewidth=0.4)
plt.show()
plt.plot(np.median(data, axis=1), linewidth=3, linestyle='--', color="black")
plt.show()