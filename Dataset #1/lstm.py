import torch
import torch.nn as nn
import torch.optim as o
from matplotlib import pyplot as plt
import time


# Params
number_of_features = 1      # input_size
number_of_classes = 3       # hidden_size
number_of_layers = 1        # num_layers

# batch_first = True
# batch - sequence - feature    => input shape
# batch - sequence - number of classes    => output shape


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(number_of_features, number_of_classes, num_layers=number_of_layers, batch_first=True)

    def forward(self, X):
        output, _ = self.lstm(X, None)
        return output[:, -1, :]


def take_data(input_path):
    data_lines = []
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
    for line in lines:
        if len(line) > 2:
            data_lines.append(line[:-1])
    data = torch.zeros((len(data_lines)), 16).float()           ## length 15
    for d, datum in enumerate(data_lines):
        splitted = datum.strip().split()
        for s, split in enumerate(splitted):
            data[d, s] = float(split)

    labels = (data[:, 0].long() - 1).reshape(data.size()[0], 1)
    data = data[:, 1:].float().reshape((data.size()[0], data.size()[1]-1, 1))

    return data, labels


def train(X, Y, model, optimizer, loss_function, device, epoch=50):

    start_time = time.process_time()
    for e in range(epoch):
        for i, data in enumerate(X):
            prediction = model(data.unsqueeze(0).to(device))
            loss = loss_function(prediction, Y[i].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if e % 10 == 0:
            print("Epoch", e, loss.detach())
    end_time = time.process_time()
    print("Training Time: ", end_time - start_time)

    return model, (end_time - start_time)


def test(X, Y, model, device):

    start_time = time.process_time()
    correct = 0
    for i, data in enumerate(X):
        prediction = model(data.unsqueeze(0).to(device))
        if torch.argmax(prediction.detach()) == Y[i]:
            correct += 1
    end_time = time.process_time()
    print("Test Time: ", end_time - start_time)
    print("Accuracy", correct/X.size()[0])

    return (correct/X.size()[0]), (end_time - start_time)


if __name__ == "__main__":
    
    train_data, train_labels = take_data("train_data.txt")
    test_data, test_labels = take_data("test_data.txt")

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
    train_accuracy_list = []
    train_set_testing_time_list = []
    test_accuracy_list = []
    test_set_testing_time_list = []

    for i in range(10):         # 10 runs
        print("Run", i+1)
        print("-----")

        m = LSTM()
        m.to(device)

        optim = o.Adam(m.parameters(), lr=0.001)
        lf = nn.CrossEntropyLoss()
        m, training_time = train(train_data, train_labels, m, optim, lf, device, epoch=400)
        training_time_list.append(training_time)

        train_acc, train_set_testing_time = test(train_data, train_labels, m, device)
        train_accuracy_list.append(train_acc)
        train_set_testing_time_list.append(train_set_testing_time)

        test_acc, test_set_testing_time = test(test_data, test_labels, m, device)
        test_accuracy_list.append(test_acc)
        test_set_testing_time_list.append(test_set_testing_time)

        print()

    print("Statistics:")
    print("Average Training Time                ----->", sum(training_time_list) / len(training_time_list))
    print("Average Training Accuracy            ----->", sum(train_accuracy_list) / len(train_accuracy_list))
    print("Maximum Training Accuracy            ----->", max(train_accuracy_list))
    print("Minimum Training Accuracy            ----->", min(train_accuracy_list))
    print("Average Testing Time of Training Set ----->", sum(train_set_testing_time_list) / len(train_set_testing_time_list))
    print("Average Testing Accuracy             ----->", sum(test_accuracy_list) / len(test_accuracy_list))
    print("Maximum Testing Accuracy             ----->", max(test_accuracy_list))
    print("Minimum Testing Accuracy             ----->", min(test_accuracy_list))
    print("Average Testing Time of Test Set     ----->", sum(test_set_testing_time_list) / len(test_set_testing_time_list))

