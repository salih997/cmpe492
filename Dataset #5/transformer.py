import torch
import torch.nn as nn
import torch.optim as o
from matplotlib import pyplot as plt
import time
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import math
from torch import Tensor
from pandas.core.common import flatten


# Constant Params
number_of_features = 1      # input_size
sequence_length = 30

# Hyperparameters
number_of_layers = 1        # num_layers
dropout = float(0.1)
pos_encode_dimension = 20   # even number
number_of_head = 1


# batch_first = True
# batch - sequence - feature    => input shape
# batch - sequence - number of classes    => output shape

## TODO 
## 1+ Encoder layer'i project ederken num feature'i duzgun project et
## 1- Add cuda to transformer model
## 2- Mnist'i test datasi olarak kullan sanity check gibi 28*28lik dataseti 28 feature, 28 sequence length gibi kullan
## 3- Multivariate olanlarin sonuclarini al
## 4- Modellerin icini iyi anla

# Note:
# nheads must divide evenly into d_model (feature dimension)


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(pos_encode_dimension, dropout, sequence_length)
        self.layers = nn.TransformerEncoderLayer(d_model=(pos_encode_dimension*number_of_features), nhead=number_of_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=number_of_layers)
        self.decoder = nn.Linear((pos_encode_dimension*number_of_features), 1)


    def forward(self, X):
        X = self.pos_encoder(X)
        encoder_output = self.transformer(X)
        output = self.decoder(encoder_output[:, -1, :])
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).reshape((1, max_len, 1))                           # [batch_size, seq_length, 1]    feature num yok bu satırda
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [(pos_encode_dimension / 2)]
        pe = torch.zeros(1, max_len, d_model)                                               # [batch_size, seq_length, pos_encode_dimension]
        pe[0, :, 0::2] = torch.sin(position * div_term)   
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x.repeat_interleave(self.pe.size()[2], dim=2) + self.pe.repeat(1,1,number_of_features)
        return self.dropout(x)


def take_data(input_path):
    df = pd.read_csv(input_path)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)
    
    data = []
    for i in range(df.values.shape[0]-sequence_length):
        data.append(df.values[i:i+sequence_length+1])
    data = torch.tensor(np.array(data), dtype=torch.float32)

    # Shuffle data
    dd = torch.randperm(data.size()[0])
    data = data[dd]

    train_data = data[:int(data.size()[0] * (4/5))]
    train_labels = train_data[:, -1, :]
    train_data = train_data[:, :-1, :]
    test_data = data[int(data.size()[0] * (4/5)):]
    test_labels = test_data[:, -1, :]
    test_data = test_data[:, :-1, :]
    
    min_value = train_data.min()
    max_value = train_data.max()
    train_data = (train_data - min_value) / (max_value - min_value)
    train_labels = (train_labels - min_value) / (max_value - min_value)
    test_data = (test_data - min_value) / (max_value - min_value)
    test_labels = (test_labels - min_value) / (max_value - min_value)
    
    return train_data, train_labels, test_data, test_labels, df


def train(X, Y, model, optimizer, loss_function, device, epoch=50):

    start_time = time.process_time()
    for e in range(1, epoch+1):
        current_loss = 0
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0).to(device))
            loss = loss_function(prediction.ravel(), Y[i].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss.item()
        # if e % 10 == 0:
        print("Epoch", e, "=> Total Loss:", current_loss)
    end_time = time.process_time()
    print("Training Time: ", end_time - start_time)
    
    return model, (end_time - start_time)


def test(X, Y, model, device):

    start_time = time.process_time()
    
    predictions = []
    for data in X:
        prediction = model(data.unsqueeze(0).to(device))
        predictions.append(prediction.ravel().tolist())
    
    r2 = r2_score(Y.detach().numpy(), predictions)
    mse = mean_squared_error(Y.detach().numpy(), predictions)
    end_time = time.process_time()
    print("Test Time: ", end_time - start_time)
    print("R2 Score: ", r2)
    print("MSE: ", mse)

    return r2, mse, (end_time - start_time), predictions


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels, data_df = take_data("data.csv")

    ##### Data Visualization #####
    
    # plt.figure(1)
    # colormap = ['b','g','r']
    # for i, data in enumerate(train_data):
    #     plt.plot(range(len(data)), data, c=colormap[train_labels[i][0]])

    # plt.figure(2)
    # for i, data in enumerate(test_data):
    #     plt.plot(range(len(data)), data, c=colormap[test_labels[i][0]])
    # plt.show()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    training_time_list = []
    train_r2_score_list = []
    train_mse_list = []
    train_set_testing_time_list = []
    test_r2_score_list = []
    test_mse_list = []
    test_set_testing_time_list = []
    a = []

    for i in range(1):         # 10 runs
        print("Run", i+1)
        print("-----")

        m = Transformer()
        m.to(device)

        optim = o.Adam(m.parameters(), lr=0.001)
        lf = nn.MSELoss()
        m, training_time = train(train_data, train_labels, m, optim, lf, device, epoch=25)
        training_time_list.append(training_time)

        train_r2_score, train_mse, train_set_testing_time, train_predictions = test(train_data, train_labels, m, device)
        a.append(train_predictions)
        train_r2_score_list.append(train_r2_score)
        train_mse_list.append(train_mse)
        train_set_testing_time_list.append(train_set_testing_time)

        test_r2_score, test_mse, test_set_testing_time, test_predictions = test(test_data, test_labels, m, device)
        a.append(test_predictions)
        test_r2_score_list.append(test_r2_score)
        test_mse_list.append(test_mse)
        test_set_testing_time_list.append(test_set_testing_time)

        print()

    print("Statistics:")
    print("Average Training Time                ----->", sum(training_time_list) / len(training_time_list))
    print("Average Training R2 Score            ----->", sum(train_r2_score_list) / len(train_r2_score_list))
    print("Average Training MSE                 ----->", sum(train_mse_list) / len(train_mse_list))
    print("Average Testing Time of Training Set ----->", sum(train_set_testing_time_list) / len(train_set_testing_time_list))
    print("Average Testing R2 Score             ----->", sum(test_r2_score_list) / len(test_r2_score_list))
    print("Average Testing MSE                  ----->", sum(test_mse_list) / len(test_mse_list))
    print("Average Testing Time of Test Set     ----->", sum(test_set_testing_time_list) / len(test_set_testing_time_list))

    plt.plot(range(data_df.values.shape[0]), data_df.values, zorder=0)
    plt.scatter(range(int((data_df.values.shape[0]-sequence_length) * (4/5))), list(flatten(a[0])), c='blue', marker='x', s=10, zorder=1)
    plt.scatter(range(int((data_df.values.shape[0]-sequence_length) * (4/5)), (data_df.values.shape[0]-sequence_length)), list(flatten(a[1])), c='tomato', marker='x', s=10, zorder=1)
    plt.show()

