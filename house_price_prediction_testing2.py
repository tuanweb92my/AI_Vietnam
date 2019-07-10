# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Commonly used modules
import numpy as np

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd

def build_model():
    model = keras.Sequential([
        Dense(20, activation=tf.nn.relu, input_shape=[13]), Dense(1)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='mse',
                  metrics=['mse'])
    return model

model = build_model()
model.load_weights('pycodes/house_price_prediction.ckpt')

(train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_housing.load_data()
train_mean = np.mean(train_features, axis=0)
train_std = np.std(train_features, axis=0)

test_features_norm = (test_features - train_mean) / train_std


test_predictions = model.predict(test_features_norm).flatten()

plt.figure()
plt.xlabel('Testing samples')
plt.ylabel('Predicted and label values')
plt.plot(np.arange(0,test_predictions.shape[0]),test_predictions, label='test_predictions')
plt.plot(np.arange(0,test_predictions.shape[0]),test_labels, label='test_labels')
plt.legend()
plt.show()
