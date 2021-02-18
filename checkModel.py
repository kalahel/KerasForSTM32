from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.python.keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
reconstructed_model = load_model("mnistSmall.h5")
X_test = X_test.astype('float32')
# Normalizing data
X_test /= 255
# one-hot encoding using keras numpy-related utilities
n_classes = 10
Y_test = np_utils.to_categorical(y_test, n_classes)
results = reconstructed_model.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)
