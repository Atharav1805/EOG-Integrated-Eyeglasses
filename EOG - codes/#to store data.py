import serial
import time
import csv
import matplotlib.pyplot as plt

# Configure the serial connection 
ser = serial.Serial('COM10', 115200)
time.sleep(2)  # Wait for the serial connection to initialize

# File to save data
filename = 'eog_data_sides_2.csv'

# Open the file for writing
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (ms)', 'Voltage (V)'])

    # Read data from serial
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                if data:
                    timestamp, voltage = map(float, data.split(','))
                    writer.writerow([timestamp, voltage])
                    print(f"Time: {timestamp} ms, Voltage: {voltage} μV")

    except KeyboardInterrupt:
        print("Data collection stopped")

# Close the serial connection
ser.close()

# Read the data from the CSV file for plotting
timestamps = []
voltages = []

with open(filename, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        timestamps.append(float(row[0]))
        voltages.append(float(row[1]))

# Plot the data
plt.plot(timestamps, voltages)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('EOG Signal')
plt.show()
