import numpy as np
import torch.nn as nn
import torch
from .utils import calculate_padding


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, kernel_size, stride=1, batch_norm=True):
        super(Deconv, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = calculate_padding(kernel_size, stride)
        self.dcl = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding // 2,
                                      bias=False)

        if batch_norm:
            self.activation = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            self.activation = nn.ReLU(True)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x):
        x = self.dcl(x,
                     output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])

        return self.activation(x)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, batch_norm=True):
        super(UnetBlock, self).__init__()
        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = calculate_padding(kernel_size, stride)
        self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=False)
        self.dcl2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding // 2, bias=False)
        if batch_norm:
            self.activation1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(True))
            self.activation2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            self.activation1 = nn.ReLU(True)
            self.activation2 = nn.ReLU(True)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, s):
        s = s.view(x.size())

        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        x = self.activation2(x)
        return x


class Generator(nn.Module):
    def __init__(self, img_size, latent_size, condition_size=0, aux_size=0, kernel_size=4, num_channels=3,
                 num_gen_channels=1024, skip_channels=[], batch_norm=True, sequential_noise=False,
                 aux_only_on_top=False):
        super(Generator, self).__init__()
        # If we have a tuple make sure we maintain the aspect ratio
        if isinstance(img_size, tuple):
            self.img_size = img_size
            self.init_size = tuple(int(4 * x / max(img_size)) for x in self.img_size)
        else:
            self.img_size = (img_size, img_size)
            self.init_size = (4, 4)

        self.latent_size = latent_size
        self.condition_size = condition_size
        self.aux_size = aux_size

        self.rnn_noise = None
        if self.aux_size > 0 and sequential_noise:
            self.rnn_noise = nn.GRU(self.aux_size, self.aux_size, batch_first=True)
            self.rnn_noise_squashing = nn.Tanh()

        self.num_layers = int(np.log2(max(self.img_size))) - 1
        self.num_channels = num_channels
        self.num_gen_channels = num_gen_channels

        self.dcl = nn.ModuleList()

        self.aux_only_on_top = aux_only_on_top
        self.total_latent_size = self.latent_size + self.condition_size

        if self.aux_size > 0 and self.aux_only_on_top:
            self.aux_dcl = nn.Sequential(
                nn.ConvTranspose2d(self.aux_size, num_gen_channels, (self.init_size[0] // 2, self.init_size[1]),
                                   bias=False),
                nn.BatchNorm2d(num_gen_channels),
                nn.ReLU(True),
                nn.ConstantPad2d((0, 0, 0, self.init_size[0] // 2), 0))
        else:
            self.total_latent_size += self.aux_size

        stride = 2
        if batch_norm:
            self.dcl.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.total_latent_size, num_gen_channels, self.init_size, bias=False),
                    nn.BatchNorm2d(num_gen_channels),
                    nn.ReLU(True)))
        else:
            self.dcl.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.total_latent_size, num_gen_channels, self.init_size, bias=False),
                    nn.ReLU(True)))

        num_input_channels = self.num_gen_channels
        in_size = self.init_size
        for i in range(self.num_layers - 2):
            if not skip_channels:
                self.dcl.append(Deconv(num_input_channels, num_input_channels // 2, in_size, kernel_size, stride=stride,
                                       batch_norm=batch_norm))
            else:
                self.dcl.append(
                    UnetBlock(num_input_channels, num_input_channels // 2, skip_channels[i], in_size,
                              kernel_size, stride=stride, batch_norm=batch_norm))

            num_input_channels //= 2
            in_size = tuple(2 * x for x in in_size)

        padding = calculate_padding(kernel_size, stride)
        self.dcl.append(nn.ConvTranspose2d(num_input_channels, self.num_channels, kernel_size,
                                           stride=stride, padding=padding // 2, bias=False))
        self.final_activation = nn.Tanh()

    def forward(self, x, c=None, aux=None, skip=[]):
        if aux is not None:
            if self.rnn_noise is not None:
                aux, h = self.rnn_noise(aux)
                aux = self.rnn_noise_squashing(aux)

            if self.aux_only_on_top:
                aux = self.aux_dcl(aux.view(-1, self.aux_size, 1, 1))
            else:
                x = torch.cat([x, aux], 2)

        if c is not None:
            x = torch.cat([x, c], 2)

        x = x.view(-1, self.total_latent_size, 1, 1)
        x = self.dcl[0](x)

        if self.aux_only_on_top:
            x = x + aux

        if not skip:
            for i in range(1, self.num_layers - 1):
                x = self.dcl[i](x)
        else:
            for i in range(1, self.num_layers - 1):
                x = self.dcl[i](x, skip[i - 1])

        x = self.dcl[-1](x, output_size=[-1, 3, self.img_size[0], self.img_size[1]])
        return self.final_activation(x)
