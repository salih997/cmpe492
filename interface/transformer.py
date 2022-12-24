import torch
import torch.nn as nn
import math
from torch import Tensor
import time

class Transformer(nn.Module):

    def __init__(self, number_of_features, number_of_classes, number_of_layers, batch_first, pos_encode_dimension, dropout, sequence_length):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(pos_encode_dimension, dropout, sequence_length)
        self.layers = nn.TransformerEncoderLayer(d_model=pos_encode_dimension, nhead=1, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=number_of_layers)
        self.decoder = nn.Linear(pos_encode_dimension, number_of_classes)


    def forward(self, X):
        X = self.pos_encoder(X)
        encoder_output = self.transformer(X)
        output = self.decoder(encoder_output[:, -1, :])
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).reshape((1, max_len, 1)) # change the latest 1 to number of features
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)   
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe
        return self.dropout(x)


def take_data_transformer(input_path, length):
    data_lines = []
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
    for line in lines:
        if len(line) > 2:
            data_lines.append(line[:-1])
    data = torch.zeros((len(data_lines)), length+1).float()
    for d, datum in enumerate(data_lines):
        splitted = datum.strip().split()
        for s, split in enumerate(splitted):
            data[d, s] = float(split)

    # Shuffle data
    dd= torch.randperm(data.size()[0])
    data = data[dd]

    labels = (data[:, 0].long() - 1).reshape(data.size()[0], 1)
    data = data[:, 1:(length+1)].float().reshape((data.size()[0], length, 1))

    return data, labels


def train_transformer(X, Y, model, optimizer, loss_function, epoch, streaming):

    start_time = time.process_time()
    for e in range(epoch):
        current_loss = 0
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0))
            loss = loss_function(prediction, Y[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss.item()
        
        streaming(f"Epoch {e} LOSS TOTAL: {current_loss}")

    end_time = time.process_time()
    streaming(f"Training Time: {end_time - start_time}")
    return model, (end_time - start_time)


def test_transformer(X, Y, model):

    start_time = time.process_time()
    correct = 0
    for i, data in enumerate(X):
        prediction = model(data.unsqueeze(0))
        if torch.argmax(prediction.detach()) == Y[i]:
            correct += 1
    end_time = time.process_time()
    
    return (correct/X.size()[0]), (end_time - start_time)
