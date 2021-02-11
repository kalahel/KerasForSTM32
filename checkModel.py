import matplotlib.pyplot as plt
import numpy as np
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
from tensorflow.python.keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()


reconstructed_model = load_model("mnistSmall.h5")
inference_result = reconstructed_model.predict(np.array([X_train[0] / 255, ]))
print(inference_result)
