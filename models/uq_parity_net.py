import os

import torch
import torch.nn as nn


# 2. Fully-connected net or Feedforward neural network
# # inputs shape: [None, dims = 2]
class Model_rho_FCNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_rho_FCNet, self).__init__()

        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self._output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, inputs):
        # split_list = [1 for i in range(2)]
        # t, x = torch.split(inputs, split_list, dim=-1)
        # operations for t, x
        # t = t / 1.0
        # x = x / 1.0
        # output = torch.cat([t, x], dim=-1)
        output = self._layer_in(inputs)
        for i, h_i in enumerate(self._hidden_layers):
            output = self.activation(h_i(output))
        output = self._output_layer(output)
        output = self.positive(output)
        return output

    def activation(self, o):
        return torch.tanh(o)

    def positive(self, o):
        return torch.exp(-o)
        # return torch.log(1.0 + torch.exp(-o))


# # inputs shape: [None, dims = 3]
class Model_r_FCNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_r_FCNet, self).__init__()

        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self._output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, inputs):
        # split_list = [1 for i in range(3)]
        # t, x, v = torch.split(inputs, split_list, dim=-1)
        # operations for t, x
        # t = t / 1.0
        # x = x / 1.0
        # v = v / 1.0
        # output = torch.cat([t, x, v], dim=-1)
        output = self._layer_in(inputs)
        for i, h_i in enumerate(self._hidden_layers):
            output = self.activation(h_i(output))
        output = self._output_layer(output)

        return output

    def activation(self, o):
        return torch.tanh(o)
        # return o*torch.sigmoid(o)


# # inputs shape: [None, dims = 3]
class Model_j_FCNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_j_FCNet, self).__init__()

        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self._output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, inputs):
        # split_list = [1 for i in range(3)]
        # t, x, v = torch.split(inputs, split_list, dim=-1)
        # operations for t, x
        # t = t / 1.0
        # x = x / 1.0
        # v = v / 1.0
        # output = torch.cat([t, x, v], dim=-1)
        output = self._layer_in(inputs)
        for i, h_i in enumerate(self._hidden_layers):
            output = self.activation(h_i(output))
        output = self._output_layer(output)

        return output

    def activation(self, o):
        # return o*torch.sigmoid(o)
        return torch.tanh(o)


# other setting


def Xavier_initi(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def save_param(net, path):
    torch.save(net.state_dict(), path)


def load_param(net, path):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        print("File does not exist.")
