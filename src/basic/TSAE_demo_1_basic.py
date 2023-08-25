import matplotlib.pyplot as plt
import numpy as np
from keras import *
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.utils import plot_model

# define input sequence
x1 = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.8, 0.6, 0.9]
x2 = [(x-0.05 + (np.power((-1), int(np.random.random(1)*10%2)) * np.random.random(1) * 0.1))[0] for x in x1]
x3 = [(x+0.05 + (np.power((-1), int(np.random.random(1)*10%2)) * np.random.random(1) * 0.1))[0] for x in x1]

sequence = np.array([x1, x2, x3])
# reshape input into [samples, timesteps, features]
num_samples, num_timesteps = sequence.shape
num_features = 1
sequence = sequence.reshape((num_samples, num_timesteps, 1))

print(sequence.shape)
print(sequence[0, :, 0])

# parameters
latent_dim = 5

# define model
model = Sequential()
model.add(LSTM(latent_dim, activation='relu', input_shape=(num_timesteps, 1)))  # i.e., (9, 1)
model.add(RepeatVector(num_timesteps))                                   # i.e., (9, 1)
model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model]
tr_epochs = 10000 # 2180
model.fit(sequence, sequence, epochs=tr_epochs, verbose=1)

plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat.shape)
print(yhat[0, :, 0])

plt.figure()

# input data
plt.plot(sequence[0, :, 0], label='x-2', color='blue')
plt.plot(sequence[1, :, 0], label='x-3', color='blue')
plt.plot(sequence[2, :, 0], label='x-3', color='blue')

# prediction
plt.plot(yhat[0, :, 0], label='y^hat-1', color='red')
plt.plot(yhat[1, :, 0], label='y^hat-2', color='red')
plt.plot(yhat[2, :, 0], label='y^hat-3', color='red')
plt.legend(loc='best')
plt.title(f'Total tr epochs = {tr_epochs}. Latent_dim = {latent_dim}')
plt.show()

