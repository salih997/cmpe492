import torch
import torch.nn as nn
import torch.optim as o


# Params
number_of_features = 1      # input_size
number_of_classes = 3       # hidden_size
number_of_layers = 1        # num_layers

# batch_first = True
# batch - sequence - feature    => input shape
# batch - sequence - number of classes    => output shape


class GRU(nn.Module):

    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(number_of_features, number_of_classes, num_layers=number_of_layers, batch_first=True)

    def forward(self, X):
        output, _ = self.gru(X, None)
        return output[:, -1, :]


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

    labels = (data[:, 0].long() - 1).reshape(data.size()[0], 1)
    data = data[:, 1:16].float().reshape((data.size()[0], 15, 1))

    return data, labels


def train(X, Y, model, optimizer, loss_function, epoch=50):

    for e in range(epoch):
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0))
            loss = loss_function(prediction, Y[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch", e, loss.detach())

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

    m = GRU()
    optim = o.Adam(m.parameters(), lr=0.001)
    lf = nn.CrossEntropyLoss()
    m = train(train_data, train_labels, m, optim, lf, epoch=400)

    test(train_data, train_labels, m)
    test(test_data, test_labels, m)
