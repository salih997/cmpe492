import torch
import torch.nn as nn
import torch.optim as o
import math
from torch import Tensor


# Constant Params
number_of_features = 1      # input_size
number_of_classes = 3       # hidden_size
sequence_length = 15

# Hyperparameters
number_of_layers = 1        # num_layers
dropout = float(0.1)
pos_encode_dimension = 10   # even number


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
        self.layers = nn.TransformerEncoderLayer(d_model=pos_encode_dimension, nhead=1, batch_first=True)
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

        position = torch.arange(max_len).reshape((1, max_len, 1))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)   
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe
        return self.dropout(x)


def take_data(input_path):
    data_lines = []
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
    for line in lines:
        if len(line) > 2:
            data_lines.append(line[:-1])
    data = torch.zeros((len(data_lines)), 16).float()           ## length 16
    for d, datum in enumerate(data_lines):
        splitted = datum.strip().split("   ")
        for s, split in enumerate(splitted):
            data[d, s] = float(split)

    # Shuffle data
    dd= torch.randperm(data.size()[0])
    data = data[dd]

    labels = (data[:, 0].long() - 1).reshape(data.size()[0], 1)
    data = data[:, 1:16].float().reshape((data.size()[0], 15, 1))

    return data, labels


def train(X, Y, model, optimizer, loss_function, epoch=50):

    for e in range(epoch):
        current_loss = 0
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0))
            loss = loss_function(prediction, Y[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = current_loss + loss.item()
        
        print("Epoch", e, "LOSS TOTAL", current_loss)
    
    return model


def test(X, Y, model):
    correct = 0
    for i, data in enumerate(X):
        prediction = model(data.unsqueeze(0))
        if torch.argmax(prediction.detach()) == Y[i]:
            correct += 1

    print("Accuracy", correct/X.size()[0])


if __name__ == "__main__":

    train_data, train_labels = take_data("train_data.txt")
    test_data, test_labels = take_data("test_data.txt")

    m = Transformer()
    optim = o.Adam(m.parameters(), lr=0.001)                           ## try SGD
    lf = nn.CrossEntropyLoss()
    m = train(train_data, train_labels, m, optim, lf, epoch=100)

    test(train_data, train_labels, m)
    test(test_data, test_labels, m)

