import torch
import torch.nn as nn
import pandas as pd

from util import *
from pruning_nn import *

result_folder = './out/result/'
model_folder = './out/model/'


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)


class DropoutNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_ratio):
        super(DropoutNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def wd():
    s = pd.DataFrame(columns=['accuracy', 'weight_decay', 'run'])

    for i in range(10):
        for w in [0.05, 0.005, 0.0005, 0.00008]:
            model = NeuralNetwork(28 * 28, 100, 10)
            acc = train_network(model, weight_decay=w)
            tmp = pd.DataFrame({
                'accuracy': [acc],
                'weight_decay': [w],
                'run': [i]
            })
            s = s.append(tmp, ignore_index=True)
        s.to_pickle('./out/result/weight-decay.pkl')


def dropout():
    s = pd.DataFrame(columns=['accuracy', 'dropout_ratio', 'run'])

    for i in range(10):
        for drop_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            model = DropoutNeuralNetwork(28*28, 100, 10, drop_ratio)

            acc = train_network(model)
            model.eval()
            tmp = pd.DataFrame({
                'accuracy': [acc],
                'dropout_ratio': [drop_ratio],
                'run': [i]
            })
            s = s.append(tmp, ignore_index=True)
        s.to_pickle('./out/result/dropout.pkl')


def train_sparse_model(filename='model'):
    s = pd.DataFrame(columns=['run', 'test_acc'])
    for i in range(10):
        model = torch.load(model_folder + filename + '.pt')
        util.reset_pruned_network(model)
        acc = train_network(model)
        s = s.append({'run': [i], 'test_acc': [acc]}, ignore_index=True)

    s.to_pickle(result_folder + filename + '-scratch.pkl')


def fine_tune_model(filename='model'):
    s = pd.DataFrame(columns=['run', 'test_acc'])
    for i in range(10):
        model = torch.load(model_folder + filename + '.pt')
        acc = train_network(model)
        s = s.append({'run': [i], 'test_acc': [acc]}, ignore_index=True)

    s.to_pickle(result_folder + filename + '-finetuned.pkl')


def train_network(model, weight_decay=0.0):
    train_set, valid_set = dataloader.get_train_valid_dataset()
    test_set = dataloader.get_test_dataset()

    # train and test the network
    lr = 0.01
    mom = 0.0
    t = True
    epoch = 0
    p_acc = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()

    while t and epoch < 200:
        model.train()
        learning.train(train_set, model, optimizer, loss_func)
        model.eval()
        new_acc = learning.test(valid_set, model)

        if new_acc - p_acc < 0.001:
            if lr > 0.0001:
                # adjust learning rate
                lr = lr * 0.1
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
            else:
                # stop training
                break

        epoch += 1
        p_acc = new_acc

    model.eval()
    return learning.test(test_set, model)


if __name__ == '__main__':
    wd()
    dropout()

    name = 'model-f'
    fine_tune_model(filename=name)
    train_sparse_model(filename=name)
