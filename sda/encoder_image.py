import numpy as np
import torch.nn as nn
from .utils import calculate_padding, is_power2

class Encoder(nn.Module):
    def __init__(self, code_size, img_size, kernel_size=4, num_input_channels=3, num_feature_maps=64, batch_norm=True):
        super(Encoder, self).__init__()

        # Get the dimension which is a power of 2
        if is_power2(max(img_size)):
            stable_dim = max(img_size)
        else:
            stable_dim = min(img_size)

        if isinstance(img_size, tuple):
            self.img_size = img_size
            self.final_size = tuple(int(4 * x // stable_dim) for x in self.img_size)
        else:
            self.img_size = (img_size, img_size)
            self.final_size = (4, 4)

        self.code_size = code_size
        self.num_feature_maps = num_feature_maps
        self.cl = nn.ModuleList()
        self.num_layers = int(np.log2(max(self.img_size))) - 2

        stride = 2
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = calculate_padding(kernel_size, stride)

        if batch_norm:
            self.cl.append(nn.Sequential(
                nn.Conv2d(num_input_channels, self.num_feature_maps, kernel_size, stride=stride, padding=padding // 2,
                          bias=False),
                nn.BatchNorm2d(self.num_feature_maps),
                nn.ReLU(True)))
        else:
            self.cl.append(nn.Sequential(
                nn.Conv2d(num_input_channels, self.num_feature_maps, kernel_size, stride=stride, padding=padding // 2,
                          bias=False),
                nn.ReLU(True)))

        self.channels = [self.num_feature_maps]
        for i in range(self.num_layers - 1):

            if batch_norm:
                self.cl.append(nn.Sequential(
                    nn.Conv2d(self.channels[-1], self.channels[-1] * 2, kernel_size, stride=stride,
                              padding=padding // 2,
                              bias=False),
                    nn.BatchNorm2d(self.channels[-1] * 2),
                    nn.ReLU(True)))
            else:
                self.cl.append(nn.Sequential(
                    nn.Conv2d(self.channels[-1], self.channels[-1] * 2, kernel_size, stride=stride,
                              padding=padding // 2, bias=False),
                    nn.ReLU(True)))

            self.channels.append(2 * self.channels[-1])

        self.cl.append(nn.Sequential(
            nn.Conv2d(self.channels[-1], code_size, self.final_size, stride=1, padding=0, bias=False),
            nn.Tanh()))

    def forward(self, x, retain_intermediate=False):
        if retain_intermediate:
            h = [x]
            for conv_layer in self.cl:
                h.append(conv_layer(h[-1]))
            return h[-1].view(-1, self.code_size), h[1:-1]
        else:
            for conv_layer in self.cl:
                x = conv_layer(x)

            return x.view(-1, self.code_size)