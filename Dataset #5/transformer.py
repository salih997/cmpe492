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


# Constant Params
number_of_features = 1      # input_size
sequence_length = 15
output_number_of_features = 1

# Hyperparameters
number_of_layers = 1        # num_layers
dropout = float(0.1)
pos_encode_dimension = 8   # even number
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=(pos_encode_dimension*number_of_features), nhead=number_of_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=number_of_layers)
        self.decoder_input_layer = nn.Linear(output_number_of_features, (pos_encode_dimension*number_of_features))
        decoder_layer = nn.TransformerDecoderLayer(d_model=(pos_encode_dimension*number_of_features), nhead=number_of_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=number_of_layers)
        self.linear_mapping = nn.Linear((pos_encode_dimension*number_of_features), output_number_of_features)


    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        encoder_output = self.encoder(src)
        tgt = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt=tgt, memory=encoder_output)
        output = self.linear_mapping(decoder_output)
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
    
    plt.plot(range(df.values.shape[0]), df.values, zorder=0)
    
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
    
    # Scale data
    train_data_min, train_data_max = train_data.min(), train_data.max()
    train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
    test_data = (test_data - train_data_min) / (train_data_max - train_data_min)
    train_labels_min, train_labels_max = train_labels.min(), train_labels.max()
    train_labels = (train_labels - train_labels_min) / (train_labels_max - train_labels_min)
    test_labels = (test_labels - train_labels_min) / (train_labels_max - train_labels_min)
    
    return train_data, train_labels, test_data, test_labels, train_labels_min, train_labels_max, dd[:train_data.shape[0]], dd[train_data.shape[0]:]


def train(X, Y, model, optimizer, loss_function, device, epoch=50):

    start_time = time.process_time()
    for e in range(1, epoch+1):
        current_loss = 0
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0).to(device), data[-1].unsqueeze(0).unsqueeze(0).to(device))
            loss = loss_function(prediction.ravel(), Y[i].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss.item()
        if e % 10 == 0:
            print("Epoch", e, "=> Total Loss:", current_loss)
    end_time = time.process_time()
    print("Training Time: ", end_time - start_time)
    
    return model, (end_time - start_time)


def test(X, Y, model, min_value, max_value, dd, plt_color, index, device):

    start_time = time.process_time()
    predictions = []
    for data in X:
        prediction = model(data.unsqueeze(0).to(device), data[-1].unsqueeze(0).unsqueeze(0).to(device))
        predictions.append(prediction.ravel().tolist())
    predictions = torch.tensor(np.array(predictions), dtype=torch.float32)
    Y = (Y * (max_value - min_value)) + min_value
    predictions = (predictions * (max_value - min_value)) + min_value
    r2 = r2_score(Y.detach().numpy(), predictions.cpu().detach().numpy())
    mse = mean_squared_error(Y.detach().numpy(), predictions.cpu().detach().numpy())
    end_time = time.process_time()
    print("Test Time: ", end_time - start_time)
    print("R2 Score: ", r2)
    print("MSE: ", mse)

    if index == 0:      # plot only the first run
        plt.scatter(dd+sequence_length, predictions.ravel().tolist(), c=plt_color, marker='x', s=10, zorder=1)

    return r2, mse, (end_time - start_time)

    
if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels, min_value, max_value, dd_train, dd_test = take_data("data.csv")

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

    for i in range(10):         # 10 runs
        print("Run", i+1)
        print("-----")

        m = Transformer()
        m.to(device)

        optim = o.Adam(m.parameters(), lr=0.001)
        lf = nn.MSELoss()
        m, training_time = train(train_data, train_labels, m, optim, lf, device, epoch=120)
        training_time_list.append(training_time)

        train_r2_score, train_mse, train_set_testing_time = test(train_data, train_labels, m, min_value, max_value, dd_train, 'blue', i, device)
        train_r2_score_list.append(train_r2_score)
        train_mse_list.append(train_mse)
        train_set_testing_time_list.append(train_set_testing_time)

        test_r2_score, test_mse, test_set_testing_time = test(test_data, test_labels, m, min_value, max_value, dd_test, 'tomato', i, device)
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

    plt.show()

