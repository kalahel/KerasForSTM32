import struct
import serial
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import time

ser = serial.Serial('COM4', 115200, timeout=20000000)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reducing the amount of data
X_test = X_test[0:int(len(X_test) / 100)]
y_test = y_test[0:int(len(y_test) / 100)]

X_test = X_test.astype('float32')

# normalizing the data from 255 to [0,1] to help with the training
X_test /= 255

# one-hot encoding using keras numpy-related utilities
n_classes = 10
Y_test = np_utils.to_categorical(y_test, n_classes)

# Sending number of entries
ser.write(struct.pack('i', len(X_test)))
rx_buffer = ser.read(4)
if struct.unpack('i', rx_buffer)[0] != len(X_test):
    print('ERROR IN ACKNOWLEDGEMENT, RECEIVED : ', struct.unpack('i', rx_buffer)[0], 'INSTEAD OF : ', len(X_test))
else:
    print('Acknowledgement valid, commencing data send')
    id = 0
    differences = []
    start_time = time.time()
    for entry_x, entry_y in zip(X_test, Y_test):

        data = np.reshape(entry_x, 28 * 28)
        # Create checksum of data
        checksum = sum(data)
        # print('Local checksum :', checksum)
        binary_data = np.array([struct.pack("f", x) for x in data])
        ser.write(binary_data)
        rx_buffer = ser.read(4)
        # Checking received checksum to monitor possible discrepancies
        if abs((checksum - struct.unpack('f', rx_buffer)[0])) > 0.01:
            print('Warning, important checksum divergence from origin :',
                  abs((checksum - struct.unpack('f', rx_buffer)[0])) * 100, '% ID : ', id)
        rx_buffer = ser.read(40)
        results = []
        # Unpacking received bytes and converting them to floats
        for i in range(0, 10):
            # Result of struct.unpack seems to be tuple hence the [0] index, referring to the desired value
            results.append(struct.unpack('f', rx_buffer[i * 4:(i + 1) * 4])[0])
        # [print(i, ' : ', x) for i, x in enumerate(results)]

        differences.append(sum(abs(np.subtract(results, entry_y))))
        id += 1
        if (id % 10 == 0):
            print(id, '/', len(X_test))
    end_time = time.time()
    print('Model mean error : ', sum(differences) / len(X_test))
    print('Total elapsed time : ', end_time - start_time, 's', '\tElapsed time per data entry : ',
          (end_time - start_time) / len(X_test), 's')
