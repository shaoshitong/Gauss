import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

"""
高斯核
"""


class SIFT_Layer(nn.Module):
    def __init__(self, img, group=1, unsample_mode="linear", sigma=1.0, double_base=True):
        """
        @param unsample_mode:='nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        @param shape:=feature,width,height
        @param kernel_size:=in_channels, out_channels // groups, *kernel_size
        @param bias:=in_channels
        """
        super(SIFT_Layer, self).__init__()
        self.unsample = nn.Upsample(None, 2, mode=unsample_mode)
        self.sigma = sigma
        self.kernel_size1 = 5
        self.kernel_size2 = 9
        self.shape = img.shape
        self.group = group
        self.doulb_base = double_base
        self.in_channel = self.shape[0]
        self.out_channel = self.in_channel
        self.gauss2D_kernel1 = self._gauss2D(self.kernel_size1, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias1 = Parameter(torch.zeros(self.in_channel))
        self.GaussianBlur1 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel1, bias=self.gauss2D_bias1, stride=1,
                                                padding=1)
        self.gauss2D_kernel2 = self._gauss2D(self.kernel_size2, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias2 = Parameter(torch.zeros(self.in_channel))
        self.GaussianBlur2 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel2, bias=self.gauss2D_bias2, stride=1,
                                                padding=1)

    def _gauss2D(self, kernel_size, sigma, group, in_channel, out_channel):
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * s))
                sum_val += kernel[i, j]
        out_channel = out_channel // group
        kernel = ((kernel / sum_val).unsqueeze(0)).unfold(dimension=0, size=out_channel, step=1)
        kernel = Parameter((kernel.unsqueeze(0)).unfold(dimension=0, size=in_channel, step=1))
        return kernel

    def _gauss1D(self, kernel_size, sigma, group, in_channel, out_channel):
        kernel = torch.zeros(kernel_size)
        center = kernel_size // 2
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            x = i - center
            kernel[i] = math.exp(-(x ** 2) / (2 * s))
            sum_val += kernel[i]
        out_channel = out_channel // group
        kernel = ((kernel / sum_val).unsqueeze(0)).unfold(dimension=0, size=out_channel, step=1)
        kernel = Parameter((kernel.unsqueeze(0)).unfold(dimension=0, size=in_channel, step=1))
        return kernel

    def forward(self, x):
        x = self.GaussianBlur1(x)
        if self.doulb_base:
            x = self.unsample(x)
        x = self.GaussianBlur2(x)
        return x
