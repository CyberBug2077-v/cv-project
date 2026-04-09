"""Learnable Gabor feature blocks used by the Gabor U-Net variants.

Parts of this implementation are adapted from the CCNet repository:
https://github.com/Zi-YuanYang/CCNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings


class GaborConv2d(nn.Module):
    """Convolution layer whose filters are generated from learnable Gabor parameters."""

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      

        self.init_ratio = init_ratio 

        self.kernel = 0

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # These defaults come from the source implementation and give a sensible
        # starting receptive field before training adapts the filters.
        self._SIGMA = 9.19237995147705 * self.init_ratio
        self._FREQ = 0.062152199447155 / self.init_ratio
        self._GAMMA = 2.0092508792877197

        # sigma and gamma control the Gaussian envelope of each filter.
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)

        # f controls the cosine frequency and psi the phase offset.
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)


    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # Build a sampling grid for every output filter.
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # Rotate each grid according to that filter's orientation.
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)

        return gb


    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

class SELayer(nn.Module):
    """Simple squeeze-and-excitation reweighting used after competition."""

    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    """Original classifier-style competition block kept for reference."""

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        self.channel_in = channel_in
        self.n_competitor = n_competitor

        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)
        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        # Small projection path used by the original CCNet block.
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2
        # print(self.weight_chan)
    def forward(self, x):

        # First-order responses.
        x = self.gabor_conv2d(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        # Second-order responses reuse the first Gabor output as input.
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0],-1),x_2.view(x_2.shape[0],-1)),dim=1)

        return xx


class FirstOrderCompetitionBlock(nn.Module):
    """
    First-order competition block for dense prediction.

    Applies a same-resolution learnable Gabor convolution, mixes channel and
    spatial competition responses, then uses SE reweighting. Unlike the
    classifier-oriented block above, this returns a feature map and does not
    use striding, pooling, or second-order competition.
    """

    def __init__(
        self,
        channel_in=3,
        n_competitor=16,
        ksize=15,
        stride=1,
        padding=7,
        weight_chan=0.5,
    ):
        super().__init__()

        self.gabor_conv2d = GaborConv2d(
            channel_in=channel_in,
            channel_out=n_competitor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
        )
        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.se = SELayer(n_competitor)

        self.weight_chan = weight_chan
        self.weight_spa = (1.0 - weight_chan) / 2.0

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x_chan = self.argmax(x)
        x_spa_x = self.argmax_x(x)
        x_spa_y = self.argmax_y(x)
        x = self.weight_chan * x_chan + self.weight_spa * (x_spa_x + x_spa_y)
        x = self.se(x)
        return x


class SecondOrderCompetitionBlock(nn.Module):
    """Return concatenated first- and second-order competition feature maps."""

    def __init__(
        self,
        channel_in=3,
        n_competitor=32,
        ksize=15,
        stride=1,
        padding=7,
        weight_chan=0.5,
    ):
        super().__init__()

        self.gabor_conv2d = GaborConv2d(
            channel_in=channel_in,
            channel_out=n_competitor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
        )
        self.gabor_conv2d2 = GaborConv2d(
            channel_in=n_competitor,
            channel_out=n_competitor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
        )
        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight_chan
        self.weight_spa = (1.0 - weight_chan) / 2.0

    def _compete(self, x, se_layer):
        # Mix channel-wise and spatial competition before SE reweighting.
        x_chan = self.argmax(x)
        x_spa_x = self.argmax_x(x)
        x_spa_y = self.argmax_y(x)
        x = self.weight_chan * x_chan + self.weight_spa * (x_spa_x + x_spa_y)
        return se_layer(x)

    def forward(self, x):
        first_order_raw = self.gabor_conv2d(x)
        first_order = self._compete(first_order_raw, self.se1)

        second_order_raw = self.gabor_conv2d2(first_order_raw)
        second_order = self._compete(second_order_raw, self.se2)

        return torch.cat([first_order, second_order], dim=1)
