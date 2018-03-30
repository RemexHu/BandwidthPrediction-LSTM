import pandas
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import random

#dataset = pandas.read_csv('train.csv', usecols=[1], engine='python')
#dataset = dataset[:1600]

filedict = {'bus': 11, 'car': 5, 'ferry': 15, 'metro': 16, 'train': 4, 'tram': 17}
file = []

for category, num in filedict.items():
    for i in range(num):
        file.append(pandas.read_csv('train_sim_traces/' + category + str(i) +'.log', names = ["band"], sep = '\t'))


"""
file.append(pandas.read_csv('train.oslo-vestby-report.2011-02-11_1618CET.log', names = ["band"], sep = '\t'))
file.append(pandas.read_csv('train_sim_traces/tram.jernbanetorget-ljabru-report.2010-12-09_1222CET.log', names = ["band"], sep = '\t'))
file.append(pandas.read_csv('train_sim_traces/ferry.nesoddtangen-oslo-report.2010-09-20_1542CEST.log', names = ["band"], sep = '\t'))
file.append(pandas.read_csv('train_sim_traces/car.aarnes-elverum-report.2011-02-10_1611CET.log', names = ["band"], sep = '\t'))
file.append(pandas.read_csv('train_sim_traces/bus.ljansbakken-oslo-report.2011-01-31_1025CET.log', names = ["band"], sep = '\t'))
file.append(pandas.read_csv('train_sim_traces/metro.kalbakken-jernbanetorget-report.2010-09-14_2303CEST.log', names = ["band"], sep = '\t'))
"""

print(len(file))
print('Training data loaded!')


flag = 0
try:
    model_mse = keras.models.load_model('model_mse_all_wnorm_simple.h5')
    print('load existing model succeed')
except:
    flag = 1


for epoch in range(10):
    print("==================== No.{epoch} epoch ====================".format(epoch=epoch + 1))

    random.shuffle(file)

    for i, cur in enumerate(file):
        print("==================== No.{epoch} epoch ====================".format(epoch=epoch + 1))
        print('===================  No.{i}/{total} trace  ================='.format(i=i+1, total=len(file)))

        cur_data = np.asarray(cur)
        cur_data = np.vstack(cur_data)

        # plt.figure(figsize=(25,9))
        # plt.plot(dataset)
        # plt.show()

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #cur_data = scaler.fit_transform(cur_data)
        """
        train_size = int(len(cur_data) * 0.9)
        test_size = len(cur_data) - train_size
        train, test = cur_data[0:train_size, :], cur_data[train_size:len(cur_data), :]
        train_rev, test_rev = train[::-1], test[::-1]
        """

        train = cur_data
        train_rev = train[::-1]


        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []

            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])

            return numpy.array(dataX), numpy.array(dataY)


        look_back = 1
        trainX, trainY = create_dataset(train, look_back)

        trainX_rev, trainY_rev = create_dataset(train_rev, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

        trainX_rev = numpy.reshape(trainX_rev, (trainX_rev.shape[0], trainX_rev.shape[1], 1))

        if flag == 1:
            print('Training from new model')
            model_mse = Sequential()
            model_mse.add(GRU(256, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
            # model_mse.add(Dropout(0.4))
            # model_mse.add(GRU(256, return_sequences=True))
            model_mse.add(Dropout(0.4))
            # model_mse.add(Dropout(0.2))
            model_mse.add(GRU(64))
            # model_mse.add(Dropout(0.2))

            model_mse.add(Dense(1))
            model_mse.compile(loss='mean_squared_error', optimizer='adam')
            flag = 0


        try:
            model_mse.fit(trainX, trainY, epochs=20, batch_size=64, verbose=2)
            model_mse.save('model_mse_all_wnorm_simple.h5')
        except KeyboardInterrupt:
            model_mse.save('model_mse_all_wnorm_simple.h5')

        try:
            model_mse.fit(trainX_rev, trainY_rev, epochs=20, batch_size=64, verbose=2)
            model_mse.save('model_mse_all_wnorm_simple.h5')
        except KeyboardInterrupt:
            model_mse.save('model_mse_all_wnorm_simple.h5')

        model_mse.save('model_mse_all_wnorm_simple.h5')



model_mse.save('model_mse_all_wnorm_simple.h5')

print('Training process completed! Model saved!')