import pandas
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np


#file = pandas.read_csv('test_sim_traces/norway_bus_1', names = ["band"], sep = '\t')
#file = pandas.read_csv('test_sim_traces/norway_tram_1', names = ["band"], sep = '\t')
#file = pandas.read_csv('test_sim_traces/norway_train_1', names = ["band"], sep = '\t')
file = pandas.read_csv('test_sim_traces/norway_car_1', names = ["band"], sep = '\t')
#file = pandas.read_csv('test_sim_traces/norway_metro_1', names = ["band"], sep = '\t')
#file = pandas.read_csv('test_sim_traces/norway_ferry_1', names = ["band"], sep = '\t')



dataset = file["band"][:]
dataset = dataset[:]

dataset = np.asarray(dataset)
dataset = np.vstack(dataset)

model_mse = keras.models.load_model('model_mse_all_norm_cmplx2.h5')
model_mse.summary()

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)

look_back = 1
testX, testY = create_dataset(dataset, look_back)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
testY_inverse = scaler.inverse_transform([testY])

testPredict = model_mse.predict(testX)
testPredict_mse = scaler.inverse_transform(testPredict)



testPredict = np.hstack(testPredict)
print(testY.shape, testPredict.shape)

testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict_mse[:,0]))
#testScore = math.sqrt(mean_squared_error(testY, testPredict))



print('Test Score: %.2f RMSE' % (testScore))



plt.figure(figsize=(25,9))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(testPredict_mse)
plt.savefig('/home/runchen/Pictures/all_norm_simple_ferry_' + str(round(testScore, 2)) +'.png')
plt.show()

