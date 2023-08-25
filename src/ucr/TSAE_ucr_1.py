import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import *
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, GRU
from keras.utils import plot_model

# define input sequence
# x1 = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.8, 0.6, 0.9]
# x2 = [(x-0.05 + (np.power((-1), int(np.random.random(1)*10%2)) * np.random.random(1) * 0.1))[0] for x in x1]
# x3 = [(x+0.05 + (np.power((-1), int(np.random.random(1)*10%2)) * np.random.random(1) * 0.1))[0] for x in x1]

ucr_dataset_dir = '/Users/michaelchan/Data_michaelychen/Workspace/python-workspace/Datasets/ucr/'
tr_dataset_path = 'Mallat/Mallat_TRAIN.tsv'
test_dataset_path = 'Mallat/Mallat_TEST.tsv'

train_data = pd.read_csv(ucr_dataset_dir + tr_dataset_path, sep='\t')
test_data = pd.read_csv(ucr_dataset_dir + test_dataset_path, sep='\t')
pass

tr_seq = train_data.values[:, 1:]
test_seq = test_data.values[:, 1:]

# np.isnan(tr_seq).any()
# np.isnan(test_seq).any()

# reshape input into [samples, timesteps, features]
num_tr_samples, num_timesteps = tr_seq.shape
num_test_samples, _ = test_seq.shape

num_features = 1
tr_sequence = tr_seq.reshape((num_tr_samples, num_timesteps, 1))
test_sequence = test_seq.reshape((num_test_samples, num_timesteps, 1))

print(tr_sequence.shape)

# parameters
tr_epochs = 1500 # 2180
latent_dim = 100
bs = 32
lr = 0.00001


# define model
model = Sequential()
model.add(GRU(latent_dim, activation='relu', input_shape=(num_timesteps, 1)))  # LSTM suffered from the NAN issue, while GRU works perfectly.
model.add(RepeatVector(num_timesteps))
model.add(GRU(latent_dim, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr))

# fit model]
st_time = time.time()
history = model.fit(tr_sequence, tr_sequence, epochs=tr_epochs, batch_size=32, verbose=1)
end_time = time.time()

plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(tr_sequence, verbose=1)


print(yhat.shape)
print(yhat[0, :, 0])

plt.figure()

# input data
plt.plot(tr_sequence[0, :, 0], label='x-1', color='blue')
# plt.plot(sequence[1, :, 0], label='x-2', color='blue')
# plt.plot(sequence[2, :, 0], label='x-3', color='blue')
# plt.plot(test_seq[0, :, 0], label='x-1', color='green')

# prediction
plt.plot(yhat[0, :, 0], label='y^hat-1', color='red')
# plt.plot(yhat[1, :, 0], label='y^hat-2', color='red')
# plt.plot(yhat[2, :, 0], label='y^hat-3', color='red')
plt.legend(loc='best')
plt.title(f'UCR - Total training epochs = {tr_epochs}. Latent_dim = {latent_dim}')
# NOTE: tr_epochs - latent_dim - lr - batch_size
plt.savefig(f'Result_{tr_epochs}_{latent_dim}_{bs}_{lr}.png')
plt.show()

print(f'Total fitting time: {end_time - st_time}')


