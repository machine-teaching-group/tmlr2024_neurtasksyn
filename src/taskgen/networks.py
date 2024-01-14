import torch
import torch.nn.functional as F
from torch import nn


class SimpleFullyConnectedNetwork(nn.Module):
    def __init__(self, input_size,
                 layer_sizes):
        super(SimpleFullyConnectedNetwork, self).__init__()
        self.input_size = input_size

        self.layers = []
        for i, layer_size in enumerate(layer_sizes):
            self.layers.append(nn.Linear(input_size, layer_size))
            # self.layers.append(nn.ReLU())
            input_size = layer_size
            self.add_module("layer_{}".format(i), self.layers[-1])
        # self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        # x = self.layers[-1](x)

        return x


class HandmadeFeaturesNetwork(nn.Module):
    def __init__(self, input_size,
                 layer_sizes,
                 output_size):
        super(HandmadeFeaturesNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.net = SimpleFullyConnectedNetwork(input_size,
                                               layer_sizes)
        self.action_layer = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        x = self.net(x)
        x = self.action_layer(x)

        return x


class ActionValueHandmadeFeaturesNetwork(nn.Module):
    def __init__(self, input_size,
                 layer_sizes,
                 output_size):
        super(ActionValueHandmadeFeaturesNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.net = SimpleFullyConnectedNetwork(input_size,
                                               layer_sizes)
        self.action_layer = nn.Linear(layer_sizes[-1], output_size)
        self.value_layer = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x):
        x = self.net(x)
        action = self.action_layer(x)
        value = self.value_layer(x)

        return action, value


"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/3
"""


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class GridCNN(nn.Module):
    def __init__(self, input_size_tuple,
                 cnn_layer_sizes=((16, 3, 1, 1, 1), (32, 3, 1, 1, 1), (64, 3, 1, 1, 1)),
                 pooling_layers=((2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0))):
        super(GridCNN, self).__init__()
        self.input_size_tuple = input_size_tuple
        self.cnn_layer_sizes = cnn_layer_sizes
        self.pooling_layers_sizes = pooling_layers

        self.conv_and_pooling_layers = []
        self.sizes = []
        for i, ((layer_size, kernel_size, dilation, stride, padding),
                (pooling_kernel_size, pooling_stride, pooling_dilation, pooling_padding)) \
                in enumerate(zip(cnn_layer_sizes, pooling_layers)):
            self.conv_and_pooling_layers.append(nn.Conv2d(input_size_tuple[0], layer_size, kernel_size=kernel_size,
                                                          dilation=dilation, stride=stride, padding=padding))
            self.conv_and_pooling_layers.append(nn.ReLU())
            self.conv_and_pooling_layers.append(nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                                             stride=pooling_stride,
                                                             dilation=pooling_dilation,
                                                             padding=pooling_padding))
            h, w = conv_output_shape(input_size_tuple[1:], kernel_size=kernel_size, stride=stride, pad=padding,
                                     dilation=dilation)
            h, w = conv_output_shape((h, w), kernel_size=pooling_kernel_size, stride=pooling_stride,
                                     pad=pooling_padding,
                                     dilation=pooling_dilation)
            input_size_tuple = (layer_size, h, w)
            self.sizes.append((layer_size, h, w))

        self.conv_and_pooling_layers = nn.Sequential(*self.conv_and_pooling_layers)

    def forward(self, x):
        for layer in self.conv_and_pooling_layers:
            x = layer(x)
        return x


class GridHandmadeFeaturesNetwork(nn.Module):
    def __init__(self, input_size,
                 cnn_layer_sizes=((16, 3, 1, 1, 1), (32, 3, 1, 1, 1), (64, 3, 1, 1, 1)),
                 pooling_layers=((2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0)),
                 latent_layer_sizes=(256, 128),
                 append_features=False):
        super(GridHandmadeFeaturesNetwork, self).__init__()
        self.input_grid_size = input_size['grid']
        # self.input_features_size = input_size['features']
        self.cnn_layer_sizes = cnn_layer_sizes
        self.pooling_layers_sizes = pooling_layers
        self.latent_layer_sizes = latent_layer_sizes

        self.grid_cnn = GridCNN(input_size_tuple=self.input_grid_size,
                                cnn_layer_sizes=cnn_layer_sizes,
                                pooling_layers=pooling_layers)
        self.grid_cnn_output_size = self.grid_cnn.sizes[-1][0] * self.grid_cnn.sizes[-1][1] * self.grid_cnn.sizes[-1][2]
        self.latent_layer = SimpleFullyConnectedNetwork(input_size=self.grid_cnn_output_size,
                                                        layer_sizes=latent_layer_sizes)
        self.final_size = latent_layer_sizes[-1]
        self.append_features = append_features
        self.latent_size = None
        if 'latent' in input_size:
            self.latent_size = input_size['latent']
            self.final_size += 16
            self.helper = SimpleFullyConnectedNetwork(input_size=self.latent_size,
                                                      layer_sizes=(16,))
        elif append_features:
            self.final_size += input_size['features']

    def forward(self, grid_tensor, features_tensor):
        # print("start forward before grid_cnn")
        a = self.grid_cnn(grid_tensor)
        b = a.view(-1, self.grid_cnn_output_size)
        c = self.latent_layer(b)
        # extend with features_tensor
        if self.latent_size is not None:
            d = self.helper(features_tensor)
            x = torch.cat([c, d], dim=1)
        elif self.append_features:
            x = torch.cat([c, features_tensor], dim=1)
        else:
            x = c

        return x


class GridActionValueHandmadeFeaturesNetwork(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 cnn_layer_sizes=((16, 3, 1, 1, 1), (32, 3, 1, 1, 1), (64, 3, 1, 1, 1)),
                 pooling_layers_sizes=((2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0)),
                 latent_layer_sizes=(256, 128),
                 append_features=True,
                 decision_layers=None,
                 batch_norm=False
                 ):
        super(GridActionValueHandmadeFeaturesNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = latent_layer_sizes
        self.net = GridHandmadeFeaturesNetwork(input_size=input_size,
                                               cnn_layer_sizes=cnn_layer_sizes,
                                               pooling_layers=pooling_layers_sizes,
                                               latent_layer_sizes=latent_layer_sizes,
                                               append_features=append_features)

        self.cnn_layer_sizes = cnn_layer_sizes
        self.pooling_layers_sizes = pooling_layers_sizes

        if decision_layers is not None:
            self.dec_layers = [nn.Linear(self.net.final_size, decision_layers[0])]
            for i in range(1, len(decision_layers)):
                self.dec_layers.append(nn.ReLU())
                self.dec_layers.append(nn.Linear(decision_layers[i - 1], decision_layers[i]))
            self.dec_layers.append(nn.ReLU())
            self.dec_layers = nn.Sequential(*self.dec_layers)
        else:
            self.dec_layers = None

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        else:
            self.batch_norm = None

        self.decision_layers = decision_layers
        self.helper_layer = nn.Linear(self.net.final_size, 32)

        self.action_layer = nn.Linear(self.net.final_size if self.dec_layers is None else decision_layers[-1]
                                                                                          + self.net.final_size,
                                      output_size)
        self.value_layer = nn.Linear(self.net.final_size if self.dec_layers is None else decision_layers[-1] +
                                                                                         self.net.final_size, 1)

    def forward(self, x):
        grid_tensor = torch.stack([dec['grid'] for dec in x])
        features_tensor = torch.stack([dec['features'] for dec in x])
        if 'latent' in x[0]:
            latent_tensor = torch.stack([dec['latent'] for dec in x])
            x = self.net(grid_tensor, latent_tensor)
            x = self.helper_layer(x)
            x = F.relu(x)
            x = torch.cat([x, features_tensor], dim=1)
            action = self.action_layer(x)
            value = self.value_layer(x)
        else:
            # print("start forward before net")
            x = self.net(grid_tensor, features_tensor)
            if self.dec_layers is not None:
                y = self.dec_layers(x)
                x = torch.cat([x, y], dim=1)
            # normalize x batch

            if self.batch_norm is not None:
                x = self.batch_norm(x)

            action = self.action_layer(x)
            value = self.value_layer(x)

        return action, value

    def get_to_save(self):
        return {'cnn_layer_sizes': self.cnn_layer_sizes,
                'pooling_layers_sizes': self.pooling_layers_sizes,
                'latent_layer_sizes': self.layer_sizes,
                'input_size': self.input_size,
                'append_features': self.net.append_features,
                'decision_layers': self.decision_layers,
                'output_size': self.output_size}


class ResActionValueNetwork(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 kernel_size=3,
                 conv_stack=(64, 64, 64),
                 fc_stack=(64, 32),
                 ):
        super(ResActionValueNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_size['grid'][0], conv_stack[0],
                      kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)),
            nn.ReLU(inplace=True)
        )
        self.net = GridEncoder(kernel_size,
                               conv_stack,
                               fc_stack,
                               input_size['grid'])

        self.action_layer = nn.Linear(fc_stack[-1] + input_size['features'], output_size)
        self.value_layer = nn.Linear(fc_stack[-1] + input_size['features'], 1)
        self.kernel_size = kernel_size
        self.conv_stack = conv_stack
        self.fc_stack = fc_stack

    def forward(self, x):
        grid_tensor = torch.stack([dec['grid'] for dec in x])
        features_tensor = torch.stack([dec['features'] for dec in x])

        grid_tensor = self.initial_conv(grid_tensor)
        x = self.net(grid_tensor)
        x = torch.cat([x, features_tensor], dim=1)
        action = self.action_layer(x)
        value = self.value_layer(x)

        return action, value

    def get_to_save(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "kernel_size": self.kernel_size,
            "conv_stack": self.conv_stack,
            "fc_stack": self.fc_stack,
        }


class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        """
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        """
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out


class GridEncoder(nn.Module):
    def __init__(self,
                 kernel_size,
                 conv_stack,
                 fc_stack,
                 img_size):
        """
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        """
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i - 1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i - 1]),
                    nn.Conv2d(conv_stack[i - 1], conv_stack[i],
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i - 1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i - 1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * img_size[-1] * img_size[-2]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i - 1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i - 1), self.fc_layers[-1])

    def forward(self, x):
        """
        x: batch_size x channels x Height x Width
        """
        # batch_size = x.size(0)

        # Convolutional part
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for the fully connected part
        x = x.view(x.size(0), -1)
        # Fully connected part
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)

        return x


class HandmadeFeaturesAndSymworldNetwork(nn.Module):
    def __init__(self,
                 kernel_size,
                 conv_stack,
                 fc_stack,
                 img_size,
                 features_size,
                 output_size):
        super(HandmadeFeaturesAndSymworldNetwork, self).__init__()
        self.output_size = output_size

        self.encoder = GridEncoder(kernel_size,
                                   conv_stack,
                                   fc_stack,
                                   img_size)

        self.net = SimpleFullyConnectedNetwork(fc_stack[-1] + features_size,
                                               output_size)

    def forward(self, x):
        features = x['features']
        symworld = x['symworld']

        symworld = self.encoder(symworld)
        x = torch.cat([features, symworld], dim=0)

        return self.net(x)
