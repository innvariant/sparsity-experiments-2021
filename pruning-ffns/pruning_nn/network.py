import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Feed-forward neural network with one hidden layer. The single layers are Prunable linear layers. In these single
    neorons or weights can be deleted and will therefore not be usable any more.
    Activation function: ReLU
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = MaskedLinearLayer(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = MaskedLinearLayer(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)


class MultiLayerNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLayerNeuralNetwork, self).__init__()
        self.fc1 = MaskedLinearLayer(input_size, hidden_size)
        self.fc2 = MaskedLinearLayer(hidden_size, hidden_size)
        self.fc3 = MaskedLinearLayer(hidden_size, hidden_size)
        self.fc4 = MaskedLinearLayer(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return self.fc4(out)


class LeNet300_100(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LeNet300_100, self).__init__()
        self.fc1 = MaskedLinearLayer(input_size, 300)
        self.fc2 = MaskedLinearLayer(300, 100)
        self.fc3 = MaskedLinearLayer(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        return self.fc3(out)


class MaskedLinearLayer(nn.Linear):
    def __init__(self, in_feature, out_features, bias=True, keep_layer_input=False):
        """
        :param in_feature:          The number of features that are inserted in the layer.
        :param out_features:        The number of features that are returned by the layer.
        :param bias:                Iff each neuron in the layer should have a bias unit as well.
        :param keep_layer_input:    Iff the Mask should also store the layer input for further calculations. This is
                                    needed by
        """
        super().__init__(in_feature, out_features, bias)
        # create a mask of ones for all weights (no element pruned at beginning)
        self.mask = Variable(torch.ones(self.weight.size()))
        self.saliency = None
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def get_saliency(self):
        if self.saliency is None:
            return self.weight.data.abs()
        else:
            return self.saliency

    def set_saliency(self, sal):
        if not sal.size() == self.weight.size():
            raise ValueError('mask must have same size as weight matrix')

        self.saliency = sal

    def get_mask(self):
        return self.mask

    def set_mask(self, mask=None):
        if mask is not None:
            self.mask = Variable(mask)
        self.weight.data = self.weight.data * self.mask.data

    def get_weight_count(self):
        return self.mask.sum()

    def get_weight(self):
        return self.weight

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        if not keep_mask:
            self.mask = Variable(torch.ones(self.weight.size()))
            self.saliency = None

    def forward(self, x):
        # eventually store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data
        weight = self.weight.mul(self.mask)
        return F.linear(x, weight, self.bias)
