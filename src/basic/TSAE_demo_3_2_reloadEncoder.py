import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import *
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.utils import plot_model


# define input sequence
x1 = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.8, 0.6, 0.9]
x2 = [(x-0.05 + (np.power((-1), int(np.random.random(1) * 10 % 2)) * np.random.random(1) * 0.1))[0] for x in x1]
x3 = [(x+0.05 + (np.power((-1), int(np.random.random(1) * 10 % 2)) * np.random.random(1) * 0.1))[0] for x in x1]

sequence = np.array([x2, x3])
test_seq = np.array([x1])
# reshape input into [samples, timesteps, features]
num_samples, num_timesteps = sequence.shape
num_features = 1
sequence = sequence.reshape((num_samples, num_timesteps, 1))
test_seq = test_seq.reshape((1, num_timesteps, 1))

# connect the encoder LSTM as the output layer
encoder = tf.keras.saving.load_model("model.keras")

# get the feature vector for the input sequence
test_encoded = encoder.predict(test_seq)
print(test_encoded.shape)

