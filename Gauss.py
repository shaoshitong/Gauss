import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.utils.data
import cv2 as cv
from ImageDataloader import ImageFilelist
from visprint import vis_img
import numpy as np

"""
高斯核
"""

"""
struct ImageOctave     /*金字塔每一阶*/


    int row, col;          //Dimensions of image.   
    float subsample;
    ImageLevel* Octave;
    
struct ImageLevel         /*金字塔阶内的每一层*/
    float levelsigma;
    int levelsigmalength;       //作用于前一张图上的高斯直径
    float absolute_sigma;
    Mat Level;


"""


class ImageLevel(object):
    levelsigma: float
    levelsigmalen: int
    absolute_sigma: float
    level: torch.Tensor

    def __init__(self, levelsigma=0., levelsigmalen=0, absolute_sigma=0., level=torch.randn(3, 32, 32)):
        self.levelsigma = levelsigma
        self.level = level
        self.levelsigmalen = levelsigmalen
        self.absolute_sigma = absolute_sigma

    def set(self, str_name, val):
        getattr(self, str_name, val)


class ImageOctave(object):
    row: int
    col: int
    subsample: float
    ImageLevel: list

    def __init__(self, row=0, col=0, subsample=0., ImageLevel=None):
        self.row = row
        self.col = col
        self.subsample = subsample
        self.ImageLevel = ImageLevel

    def set(self, str_name, val):
        getattr(self, str_name, val)


class SIFT_Layer(nn.Module):
    def __init__(self, img, group=1, unsample_mode="bilinear", sigma=1.0, double_base=True, Scale=2):
        """
        @param unsample_mode:='nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        @param shape:=feature,width,height
        @param kernel_size:=in_channels, out_channels // groups, *kernel_size
        @param bias:=in_channels
        """
        super(SIFT_Layer, self).__init__()
        self.unsample = nn.Upsample(None, scale_factor=2, mode=unsample_mode, align_corners=True)
        self.sigma = sigma
        self.kernel_size1 = 5
        self.kernel_size2 = 9
        self.shape = img.shape
        self.group = group
        self.doulb_base = double_base
        self.in_channel = self.shape[0]
        self.out_channel = self.in_channel
        """
        double k = pow(2, 1.0 / ((float)SCALESPEROCTAVE));  /
        int num_peroc_levels = c + 3;
        int num_perdog_levels = num_peroc_levels - 1;
        """
        self.npcl = Scale + 3
        self.npdl = self.npcl - 1
        self.float_sigma = math.sqrt(2.)
        self.float_k = math.pow(2, 1. / self.npcl)
        self.gauss2D_kernel1 = self._gauss2D(self.kernel_size1, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias1 = Parameter(torch.zeros(self.in_channel), requires_grad=False)
        print(self.gauss2D_bias1.shape, self.gauss2D_kernel1.shape)
        self.GaussianBlur1 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel1, bias=self.gauss2D_bias1, stride=1,
                                                groups=self.group,
                                                padding=1)
        self.gauss2D_kernel2 = self._gauss2D(self.kernel_size2, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias2 = Parameter(torch.zeros(self.in_channel), requires_grad=False)
        self.GaussianBlur2 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel2, bias=self.gauss2D_bias2, stride=1,
                                                groups=self.group,
                                                padding=1)

    """
    TODO : found
    """

    def _found_Octaves_parameter(self, ):
        nums = self._get_nums()
        octaves = {}
        DOGoctaves = {}
        for num in range(nums):
            octaves[num + 1] = ImageOctave()
            DOGoctaves[num + 1] = ImageOctave()
            octaves[num + 1].ImageLevel = [ImageLevel for i in range(self.npcl)]
            octaves[num + 1].ImageLevel = [ImageLevel for i in range(self.npdl)]
            octaves[num + 1].col = self.shape[-2], octaves[num + 1].row = self.shape[-1]
            DOGoctaves[num + 1].col = self.shape[-2], DOGoctaves[num + 1].row = self.shape[-1]
            octaves[num + 1].subsample = math.pow(2, num) / 2. if self.doulb_base else octaves[
                num + 1].subsample = math.pow(2, num)

    """
    get number
            int numoctaves = 4;

        {
            int dim = min(init_Mat.rows, init_Mat.cols);
            numoctaves = (int)(log((double)dim) / log(2.0)) - 2;    //金字塔阶数
            numoctaves = min(numoctaves, MAXOCTAVES);
            sf->numoctaves = numoctaves;
        }

    """

    def _get_nums(self):
        shape = self.shape
        dim = min(shape[-2], shape[-1])
        nums = (int)(math.log(dim) / math.log(2.0)) - 2
        nums = min(nums, 4)
        return nums

    def get_uss2D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False):
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
        kernel = ((kernel / sum_val).unsqueeze(0)).repeat(out_channel, 1, 1)
        kernel = Parameter((kernel.unsqueeze(0)).repeat(in_channel, 1, 1, 1), requires_grad == requires_grad)
        return kernel

    def _gauss1D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False):
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
        kernel = ((kernel / sum_val).unsqueeze(0)).repeat(out_channel, 1)
        kernel = Parameter((kernel.unsqueeze(0)).repeat(in_channel, 1, 1), requires_grad == requires_grad)
        return kernel

    def forward(self, x):
        x = self.GaussianBlur1(x)
        if self.doulb_base:
            x = self.unsample(x)
        x = self.GaussianBlur2(x)
        return x


# mm=nn.Conv2d(2,4,groups=2,kernel_size=3,stride=1,padding=1)
# print(list(mm.parameters())[0].shape,"aaa")
# print(mm.forward(torch.randn(5,2,3,3)))
if __name__ == "__main__":
    img = torch.randn(3, 32, 32)
    sift_layer = SIFT_Layer(img, group=3)
    a = ImageFilelist("tmp.txt", transform=None)
    test_loader = torch.utils.data.DataLoader(a, batch_size=64,
                                              shuffle=True, num_workers=4)
    for i, (image, label) in enumerate(test_loader):
        if i > 1:
            break
        output = sift_layer.forward(image)[0].permute(2, 1, 0)
        vis_img([image[0].permute(2, 1, 0), output])

    pass
