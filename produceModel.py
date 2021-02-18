import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils

# Loading mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Plot dataset first 9 entries
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: " + str(y_train[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data from 255 to [0,1] to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

inputs = Input(shape=(28, 28, 1))

x = Conv2D(8, (5, 5), input_shape=(28, 28, 1), use_bias=False, padding="same", kernel_regularizer=l2(0.0001))(inputs)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Conv2D(4, (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(0.0001))(x)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, "relu")(x)
x = Dense(10, "sigmoid")(x)
model = Model(inputs, x, name="mnistSmall")
# compiling the sequential model
model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
                    batch_size=128, epochs=30,
                    validation_data=(X_test, Y_test))

model.save('./mnistSmall.h5')
