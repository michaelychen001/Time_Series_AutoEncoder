import matplotlib.pyplot as plt
import numpy as np
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

# define encoder
visible = Input(shape=(num_timesteps, 1))
encoder = LSTM(100, activation='relu')(visible)

# define decoder1, that is used for reconstruction.
decoder1 = RepeatVector(num_timesteps)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)

# define decoder2, that is used for prediction.
decoder2 = RepeatVector(num_timesteps)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)

# tie it together and compile the model with defined optimizer and cost function.
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')

# fit model
tr_epochs = 2200 # 2180
model.fit(sequence, sequence, epochs=tr_epochs, verbose=1)

plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder_2.png')
# demonstrate recreation
y_hat = model.predict(test_seq, verbose=1)
yhat_recont = y_hat[0]
yhat_pred = y_hat[1]
print(yhat_recont.shape)
print(yhat_recont[0, :, 0])
print(yhat_pred[0, :, 0])

plt.figure()

# input data
plt.plot(sequence[0, :, 0], label='x-2', color='blue')
plt.plot(sequence[1, :, 0], label='x-3', color='blue')
# plt.plot(sequence[2, :, 0], label='x-3', color='blue')
plt.plot(test_seq[0, :, 0], label='x-1', color='green')

# prediction
plt.plot(yhat_recont[0, :, 0], label='y^hat-1-reconstruction', color='red')
plt.plot(yhat_pred[0, :, 0], label='y^hat-1-prediction', color='orange')
# plt.plot(yhat[1, :, 0], label='y^hat-2', color='red')
# plt.plot(yhat[2, :, 0], label='y^hat-3', color='red')
plt.legend(loc='best')
plt.title(f'Total tr epochs = {tr_epochs}')
plt.show()

