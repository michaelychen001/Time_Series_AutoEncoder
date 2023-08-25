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

print(sequence.shape)
print(sequence[0, :, 0])

# define model
model = Sequential()
model.add(LSTM(3, activation='relu', input_shape=(num_timesteps, 1)))
model.add(RepeatVector(num_timesteps))
model.add(LSTM(3, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# setting up saving checkpoints
checkpoint_filepath = '/tmp/checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_freq=50,
    save_weights_only=True,
    monitor='loss', # 'val_accuracy'
    mode='max',
    save_best_only=True)


# fit model
tr_epochs = 300 # 2180
model.fit(sequence, sequence, callbacks=[model_checkpoint_callback],
          epochs=tr_epochs, verbose=1)

# connect the encoder LSTM as the output layer
encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)
# save the encoder to file
encoder.save("encoder_model.keras")

plot_model(encoder, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
test_encoded = encoder.predict(test_seq)
print(test_encoded.shape)

