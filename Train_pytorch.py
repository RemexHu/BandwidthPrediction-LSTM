import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pandas

TIME_STEP = 20
TARGET_SIZE = 1
INPUT_SIZE = 1
HIDDEN_SIZE = 128

BATCH_SIZE = 10
LR = 0.001
EPOCH = 20

def create_Dataset(dir):
    # Input: dir, path of log files
    # Output: dataset, flattened numpy array
    file = pandas.read_csv(dir, names = ["band"], sep = '\t')
    dataset = file["band"][:]
    dataset = np.asarray(dataset).astype(float)
    return dataset

def create_DataTensor(dataset, window):
    # Input: dataset, flattened numpy array
    # Output: Tensor_x [batch, time_step, features]
    #         Tensor_y [batch, target_size, features]
    # Each dataset has one output
    data_x, data_y = [], []
    for i in range(window, len(dataset)):
        batch_x = dataset[i - window: i]
        batch_y = dataset[i]
        data_x.append(batch_x[:, np.newaxis])
        data_y.append(np.array([[batch_y]]))

    data_x, data_y = np.asarray(data_x), np.asarray(data_y)

    Tensor_x = torch.from_numpy(data_x)
    Tensor_y = torch.from_numpy(data_y)

    return Tensor_x, Tensor_y

def create_DataLoader(Tensor_x, Tensor_y):
    Tensor_batch = Data.TensorDataset(data_tensor=Tensor_x, target_tensor=Tensor_y)
    loader = Data.DataLoader(dataset=Tensor_batch, batch_size=BATCH_SIZE, drop_last=True)

    return loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.out = nn.Linear(HIDDEN_SIZE * TIME_STEP, TARGET_SIZE)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        # r_out shape (batch, time_step, hidden_size)
        # then do flattening in order to pass through the linear layer
        r_out = r_out.contiguous().view(BATCH_SIZE, 1, HIDDEN_SIZE * TIME_STEP)
        # print('1')
        # print(r_out.size())
        outs = self.out(r_out)
        # print(outs.size())
        return outs


def train(model, DataLoader_train, DataLoader_test, epochs, optimizer, loss_fn):
    loss_curve = []

    for epoch in range(epochs):
        itr = 0
        for loader in DataLoader_train:
            for (batch_x, batch_y) in loader:
                batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
                x, y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                output = model(x)

                loss = loss_fn(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                itr += 1

                if itr % 200 == 0:
                    loss_val = val(model=model, DataLoader_test=DataLoader_test, loss_fn=loss_fn)
                    loss_curve.append(loss_val)
                    print('Epoch{} Iter{} val_oss{}'.format(epoch, itr, loss_val))


    loss_array = np.asarray(loss_curve)
    loss_array = np.hstack(loss_array)

    return model, loss_array


def val(model, DataLoader_test, loss_fn):
    loss_itr = []
    for loader in DataLoader_test:
        for (batch_x, batch_y) in loader:
            batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
            x, y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            output = model(x)
            loss_itr.append(loss_fn(output, y))

    return np.sum(sum(loss_itr).data.cpu().numpy())



def main():

    filedict_train = {'bus': 11, 'car': 5, 'ferry': 15, 'metro': 16, 'train': 4, 'tram': 17}
    training = []
    for category, num in filedict_train.items():
        for i in range(num):
            dir = 'train_sim_traces/' + category + str(i) +'.log'
            training.append(create_Dataset(dir))

    filedict_test = {'bus': 2, 'car': 2, 'ferry': 2, 'metro': 2, 'train': 2, 'tram': 2}

    test = []
    for category, num in filedict_test.items():
        for i in range(num):
            dir = 'test_sim_traces/norway_' + category + '_' + str(i + 1)
            test.append(create_Dataset(dir))



    DataLoader_train, DataLoader_test = [], []
    for dataset in training:
        Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
        loader = create_DataLoader(Tensor_x, Tensor_y)
        DataLoader_train.append(loader)
    for dataset in test:
        Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
        loader = create_DataLoader(Tensor_x, Tensor_y)
        DataLoader_test.append(loader)


    BandwidthLSTM = Net()
    BandwidthLSTM.cuda()
    optimizer = optim.Adam(BandwidthLSTM.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train(model=BandwidthLSTM,
          DataLoader_train=DataLoader_train,
          DataLoader_test=DataLoader_test,
          epochs=EPOCH,
          optimizer=optimizer,
          loss_fn=loss_fn)

    torch.save(BandwidthLSTM, 'net1.pkl')

if __name__ == '__main__':
    main()


