import pandas as pd
filename='D:\\Project data\\currentlyWorking\\7 Sufficient Data\\Oil\\Andhra Pradesh\\Groundnut\\Anantapur.csv'

df = pd.read_csv(filename) 
ff=df
ff=ff.iloc[:,1:]

ff=ff[ff['arrivalquantity']<2]

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import StandardScaler

#label encoding
def category(row):
    if row=='rain':
        return 1
    return 0

ff['weather']=ff['weather'].apply(category)
#taking care of NaN
ff['T']=ff['T'].fillna('ffill')
ff['arrivalquantity']=ff['arrivalquantity'].fillna(0.0)
ff['MODALPRICE']=ff['MODALPRICE'].fillna('ffill')


#For Time series conversion
ff=ff.set_index('ArrivalDate')

#Transform the problem to supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#conversion from Dataframe to numpy
values = ff.values
values = values.astype('float32')


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)

reframed.head()

#Selecting only the desired parameters
reframed=reframed.iloc[:,:5]
reframed.head()


values = reframed.values

#division into training n testing sets
days = int(1989*0.8)
train = values[:days, :]
test = values[days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


import matplotlib.pyplot as pyplot

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]



test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


pyplot.plot(ff.index[days+1:],inv_yhat,'blue',label='forecasted')
pyplot.plot(ff.index[days+1:],ff.MODALPRICE[days+1:],'red',label='actual')
pyplot.legend()
pyplot.xlabel('time')
pyplot.ylabel('price')

