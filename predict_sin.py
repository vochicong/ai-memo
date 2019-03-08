# %% [markdown]
# # Sin曲線を学習してみる
#
# Sin曲線は [-1,1] の間で規則的に綺麗にカーブを描くので、その曲線の十分な長さの部分から次の値を当てることができるでしょう。

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k


# %%
# Sin曲線の1反復あたりのサンプル数
samples_per_cycle = 20 * 10
# 生成するSinサイクル数
number_of_cycles = 10
data_len = samples_per_cycle * number_of_cycles
arange = np.arange(data_len) * 2 * np.pi / samples_per_cycle
X = np.sin(arange) * .5 + .5
y = np.cos(arange) * .5 + .5 + \
    np.random.random_sample((data_len,)) * 0.05
pd.DataFrame({'X': X, 'y': y}).head(500).plot()
print(X.shape, y.shape)
xdim = 1
ydim = 1
X = X.reshape([-1, xdim])
y = y.reshape([-1, ydim])
print(X.shape, y.shape)

seq_len = samples_per_cycle // 10
batch_size = 128
n_epochs = 30
lstm_n_units = 300


def print_params():
  print('''
data_len={}
seq_len={}
batch_size={}
xdim={}
ydim={}
lstm_n_units={}
n_epochs={}
'''.format(data_len, seq_len, batch_size, xdim, ydim,
           lstm_n_units, n_epochs))


# %% [markdown]
# # Keras
# %%
kgen = k.preprocessing.sequence.TimeseriesGenerator(
    X, y, length=seq_len, batch_size=batch_size)
print(kgen[0][0].shape, kgen[0][1].shape)
assert np.array_equal(kgen[0][0][0], X[0:seq_len])
assert np.array_equal(kgen[0][0][1], X[1:seq_len + 1])
assert np.array_equal(kgen[0][1], y[seq_len:seq_len + batch_size])

# %%
kmodel = k.models.Sequential([
    k.layers.LSTM(lstm_n_units, input_shape=(seq_len, xdim,)),
    k.layers.Dense(ydim), k.layers.Activation('sigmoid'),
])
kmodel.compile(optimizer='rmsprop', loss='mean_squared_error')
print(kmodel.summary())
early_stop = k.callbacks.EarlyStopping(
    monitor='loss',
    patience=4,
    restore_best_weights=True,
    verbose=1)
reduce_lr = k.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.2,
    patience=early_stop.patience // 2,
    verbose=1)
callbacks = [reduce_lr, early_stop]
print_params()
kmodel.fit_generator(kgen, epochs=n_epochs, callbacks=callbacks)


# %%
# 学習結果と実際値の比較
pred = kmodel.predict_generator(kgen).flatten()
pd.DataFrame({'actual': y.flatten()[seq_len:],
              'predic': pred.flatten()}).head(
    samples_per_cycle * 2).plot()
pred.shape, y.flatten()[seq_len:].shape


# %% [markdown]
# # ReNom
# %%
# from renom.utility.distributor import NdarrayDistributor, TimeSeriesDistributor
# from renom.utility.trainer import Trainer
import renom as rm
rm.cuda.set_cuda_active(False)

# TODO: ReNome 時系列データ generator を作成
print(X.shape, y.shape)
print_params()
rgen = rm.utility.distributor.TimeSeriesDistributor(X, y)

# rX = X.reshape([-1, 11])
# ry = y[:rX.shape[0]]
# print(rX.shape, ry.shape)
# rgen = rm.utility.distributor.NdarrayDistributor(rX, ry)
# len(rgen), X[0], y[0], rgen[0]

# TODO: LSTMの input_size を正しく指定！
lstm = rm.Lstm(lstm_n_units,
               input_size=(seq_len, xdim,)
               # input_size=(1, xdim,)
               )
# lstm.values()
rmodel = rm.Sequential([
    # rm.Dense(300), rm.Relu(),
    lstm,
    rm.Dense(ydim), rm.Sigmoid(),
])
rtrainer = rm.utility.trainer.Trainer(
    rmodel, batch_size=batch_size, num_epoch=n_epochs,
    loss_func=rm.mean_squared_error, optimizer=rm.Rmsprop())
rtrainer.train(train_distributor=rgen)
# %%
# batch = rgen.batch(batch_size)
# lst = list(batch)
# print(len(lst[0][1]))
# lst[0][1]
# np.array(lst).shape
