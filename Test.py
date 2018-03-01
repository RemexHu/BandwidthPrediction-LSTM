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


file = pandas.read_csv('train.oslo-vestby-report.2011-02-14_0644CET.log', names = ["band"], sep = '\t')
#file = pandas.read_csv('train.oslo-vestby-report.2011-02-11_1618CET.log', names = ["band"], sep = '\t')
#file = pandas.read_csv('train_sim_traces/tram.jernbanetorget-ljabru-report.2010-12-09_1222CET.log', names = ["band"], sep = '\t')
#file = pandas.read_csv('train_sim_traces/ferry.nesoddtangen-oslo-report.2010-09-20_1542CEST.log', names = ["band"], sep = '\t')
#file = pandas.read_csv('train_sim_traces/car.aarnes-elverum-report.2011-02-10_1611CET.log', names = ["band"], sep = '\t')
#file = pandas.read_csv('test_sim_traces/norway_tram_8', names = ["band"], sep = '\t')
#file = pandas.read_csv('train_sim_traces/metro.kalbakken-jernbanetorget-report.2010-09-14_2303CEST.log', names = ["band"], sep = '\t')




dataset = file["band"][:]
dataset = dataset[:]

dataset = np.asarray(dataset)
dataset = np.vstack(dataset)

model_mse = keras.models.load_model('model_mse_uni.h5')

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

testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict_mse[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


plt.figure(figsize=(25,9))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(testPredict_mse)
plt.show()