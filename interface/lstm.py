import torch
import torch.nn as nn
import time

class LSTM(nn.Module):

    def __init__(self, number_of_features, number_of_classes, number_of_layers, batch_first):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(number_of_features, number_of_classes, num_layers=number_of_layers, batch_first=batch_first)

    def forward(self, X):
        output, _ = self.lstm(X, None)
        return output[:, -1, :]


def take_data_lstm(input_path, length):
    data_lines = []
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
    for line in lines:
        if len(line) > 2:
            data_lines.append(line[:-1])
    data = torch.zeros((len(data_lines)), length+1).float()           ## length 15
    for d, datum in enumerate(data_lines):
        splitted = datum.strip().split()
        for s, split in enumerate(splitted):
            data[d, s] = float(split)

    labels = (data[:, 0].long() - 1).reshape(data.size()[0], 1)
    data = data[:, 1:].float().reshape((data.size()[0], data.size()[1]-1, 1))

    return data, labels


def train_lstm(X, Y, model, optimizer, loss_function, device, epoch, streaming):

    start_time = time.process_time()
    for e in range(1, epoch+1):
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0).to(device))
            loss = loss_function(prediction, Y[i].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if e % 5 == 0:
            streaming(f"Epoch {e}: {loss.detach()}")
    end_time = time.process_time()
    streaming(f"Training Time: {end_time - start_time}")

    return model, (end_time - start_time)


def test_lstm(X, Y, model, device):

    start_time = time.process_time()
    correct = 0
    for i, data in enumerate(X):
        prediction = model(data.unsqueeze(0).to(device))
        if torch.argmax(prediction.detach()) == Y[i]:
            correct += 1
    end_time = time.process_time()

    return (correct/X.size()[0]), (end_time - start_time)
