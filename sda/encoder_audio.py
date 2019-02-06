import torch.nn as nn
import math
from .utils import calculate_padding, prime_factors, calculate_output_size

class Encoder(nn.Module):
    def __init__(self, code_size, rate, feat_length, init_kernel=None, init_stride=None, num_feature_maps=16,
                 increasing_stride=True):
        super(Encoder, self).__init__()

        self.code_size = code_size
        self.cl = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.strides = []
        self.kernels = []

        features = feat_length * rate
        strides = prime_factors(features)
        kernels = [2 * s for s in strides]

        if init_kernel is not None and init_stride is not None:
            self.strides.append(int(init_stride * rate))
            self.kernels.append(int(init_kernel * rate))
            padding = calculate_padding(init_kernel * rate, stride=init_stride * rate, in_size=features)
            init_features = calculate_output_size(features, init_kernel * rate, stride=init_stride * rate,
                                                        padding=padding)
            strides = prime_factors(init_features)
            kernels = [2 * s for s in strides]

        if not increasing_stride:
            strides.reverse()
            kernels.reverse()

        self.strides.extend(strides)
        self.kernels.extend(kernels)

        for i in range(len(self.strides) - 1):
            padding = calculate_padding(self.kernels[i], stride=self.strides[i], in_size=features)
            features = calculate_output_size(features, self.kernels[i], stride=self.strides[i], padding=padding)
            pad = int(math.ceil(padding / 2.0))

            if i == 0:
                self.cl.append(
                    nn.Conv1d(1, num_feature_maps, self.kernels[i], stride=self.strides[i], padding=pad))
                self.activations.append(nn.Sequential(nn.BatchNorm1d(num_feature_maps), nn.ReLU(True)))
            else:
                self.cl.append(nn.Conv1d(num_feature_maps, 2 * num_feature_maps, self.kernels[i],
                                         stride=self.strides[i], padding=pad))
                self.activations.append(nn.Sequential(nn.BatchNorm1d(2 * num_feature_maps), nn.ReLU(True)))

                num_feature_maps *= 2

        self.cl.append(nn.Conv1d(num_feature_maps, self.code_size, features))
        self.activations.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.strides)):
            x = self.cl[i](x)
            x = self.activations[i](x)

        return x.squeeze()
