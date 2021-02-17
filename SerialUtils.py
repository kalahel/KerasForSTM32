import struct
import serial
import numpy as np
from keras.datasets import mnist

# Loading mnist dataset
ser = serial.Serial('COM4', 115200, timeout=20000000)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

data = np.reshape([X_train[0] / 255], 28 * 28)
# Create checksum of data
checksum = sum(data)
print('Local checksum :', checksum)
binary_data = np.array([struct.pack("f", x) for x in data])
ser.write(binary_data)
rx_buffer = ser.read(4)
# Checking received checksum to monitor possible discrepancies
print('Received checksum :', struct.unpack('f', rx_buffer)[0], ' divergence from origin :',
      abs((checksum - struct.unpack('f', rx_buffer)[0])) * 100, '%')
rx_buffer = ser.read(40)
results = []
# Unpacking received bytes and converting them to floats
for i in range(0, 10):
    # Result of struct.unpack seems to be tuple hence the [0] index, referring to the desired value
    results.append(struct.unpack('f', rx_buffer[i * 4:(i + 1) * 4])[0])
[print(i, ' : ', x) for i, x in enumerate(results)]
