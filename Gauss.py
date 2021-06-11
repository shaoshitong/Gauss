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
        self.Scale = Scale
        self.sigma = sigma
        self.kernel_size1 = int(max(3, 2. * self.Scale + 1) // 1) + 1
        self.kernel_size2 = int(max(3, 2. * self.Scale * 2. + 1) // 1) + 1
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
        self.unsample = nn.Upsample(None, scale_factor=2, mode=unsample_mode, align_corners=True)
        self.sample = lambda x: F.interpolate(x, scale_factor=0.5, mode=unsample_mode, align_corners=True)
        self.gauss2D_kernel1 = self._gauss2D(self.kernel_size1, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias1 = Parameter(torch.zeros(self.in_channel), requires_grad=False)
        self.GaussianBlur1 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel1, bias=self.gauss2D_bias1, stride=1,
                                                groups=self.group,
                                                padding=1)
        self.gauss2D_kernel2 = self._gauss2D(self.kernel_size2, self.sigma, self.group, self.in_channel,
                                             self.out_channel)
        self.gauss2D_bias2 = Parameter(torch.zeros(self.in_channel), requires_grad=False)
        self.GaussianBlur2 = lambda x: F.conv2d(x, weight=self.gauss2D_kernel2, bias=self.gauss2D_bias2, stride=1,
                                                groups=self.group,
                                                padding=1)
        self.octaves, self.DOGoctaves = self._found_Octaves_parameter()

    """
    TODO : found
    """

    def _found_Octaves_parameter(self):
        nums = self._get_nums()
        self.nums = nums
        octaves = {}
        DOGoctaves = {}
        shape = [*(self.shape)]
        for num in range(nums):
            octaves[num + 1] = ImageOctave()
            DOGoctaves[num + 1] = ImageOctave()
            octaves[num + 1].ImageLevel = [ImageLevel() for i in range(self.npcl)]
            DOGoctaves[num + 1].ImageLevel = [ImageLevel() for i in range(self.npdl)]
            octaves[num + 1].col = int(shape[-2])
            octaves[num + 1].row = int(shape[-1])
            DOGoctaves[num + 1].col = int(shape[-2])
            DOGoctaves[num + 1].row = int(shape[-1])
            octaves[num + 1].subsample = (math.pow(2, num) / 2.) if self.doulb_base else math.pow(2, num)
            if num == 0:
                octaves[num + 1].ImageLevel[0].levelsigma = self.float_sigma
                octaves[num + 1].ImageLevel[0].absolute_sigma = self.float_sigma / 2
            else:
                octaves[num + 1].ImageLevel[0].levelsigma = self.float_sigma
                octaves[num + 1].ImageLevel[0].absolute_sigma = octaves[num].ImageLevel[self.npcl - 3].absolute_sigma
            sigma = self.float_sigma
            """
              dst = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//用于存储高斯层  
                temp = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//用于存储DOG层  

                sigma_act = sqrt(k * k - 1) * sigma;
                sigma = k * sigma;

                (octaves[i].Octave)[j].levelsigma = sigma;
                (octaves[i].Octave)[j].absolute_sigma = sigma * (octaves[i].subsample);
                // (octaves[i].Octave)[j].absolute_sigma = k *((octaves[i].Octave)[j-1].absolute_sigma);

                //产生高斯层  
                int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * sigma_act + 1.0);//高斯核的尺寸
                gaussdim = 2 * (gaussdim / 2) + 1;
                GaussianBlur((octaves[i].Octave)[j - 1].Level, dst, Size(gaussdim, gaussdim), sigma_act);
                //BlurImage((octaves[i].Octave)[j - 1].Level, dst, sigma_act);
                (octaves[i].Octave)[j].levelsigmalength = gaussdim;
                (octaves[i].Octave)[j].Level = dst;

                //产生DOG层  
                temp = ((octaves[i].Octave)[j]).Level - ((octaves[i].Octave)[j - 1]).Level;
                //subtract(((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j - 1]).Level, temp, 0);
                ((DOGoctaves[i].Octave)[j - 1]).Level = temp;
            """
            for j in range(1, self.Scale + 3):
                """
                init output
                name dst temp
                """
                sigma_act = math.sqrt(self.float_k ** 2 - 1) * self.float_sigma
                sigma = self.float_k * sigma
                octaves[num + 1].ImageLevel[j].levelsigma = sigma
                octaves[num + 1].ImageLevel[j].absolute_sigma = self.float_k * (
                    octaves[num + 1].ImageLevel[j - 1].absolute_sigma)
                gau_kernel_size = int(max(3., 2. * 3.5 * sigma_act + 1)) + 1
                octaves[num + 1].ImageLevel[j].levelsigmalen = gau_kernel_size
                ##Gauss
                ##DOG层

            ##tempMat = halfSizeImage(((octaves[i].Octave)[SCALESPEROCTAVE].Level));
            shape[-2] = shape[-2] // 2
            shape[-1] = shape[-1] // 2
        return octaves, DOGoctaves

    def f_forward(self, x):
        """
        this forward use to create DOG_image and image
        """
        octaves, DOGoctaves = self.octaves, self.DOGoctaves
        """
        (octaves[i].Octave)[0].Level = tempMat;
        """
        for num in range(self.nums):
            octaves[num + 1].ImageLevel[0].level = x.clone().detach().cpu()
            for j in range(1, self.Scale + 3):
                gauss2D_kernel = self._gauss2D(octaves[num + 1].ImageLevel[j].levelsigmalen,
                                               octaves[num + 1].ImageLevel[j].levelsigma / self.float_k, self.group,
                                               self.in_channel,
                                               self.out_channel)
                gauss2D_bias = Parameter(torch.zeros(self.in_channel), requires_grad=False)
                x = F.conv2d(x, weight=gauss2D_kernel, bias=gauss2D_bias, stride=1,
                             groups=self.group,
                             padding=(octaves[num + 1].ImageLevel[j].levelsigmalen - 1) // 2)

                octaves[num + 1].ImageLevel[j].level = x.clone().detach().cpu()
                """
                temp = ((octaves[i].Octave)[j]).Level - ((octaves[i].Octave)[j - 1]).Level;
                """

                DOGoctaves[num + 1].ImageLevel[j - 1].level = octaves[num + 1].ImageLevel[j].level - \
                                                              octaves[num + 1].ImageLevel[j - 1].level
                print(DOGoctaves[num + 1].ImageLevel[j - 1].level[0, :, 0, 0])
            x = self.sample(x)
        return x

    """
            float sigma = init_sigma;
            float sigma_act, absolute_sigma;    //每次直接作用于前图像上的blur值    ；   尺度空间中的绝对值
    """

    def _get_nums(self):
        shape = self.shape
        dim = min(shape[-2], shape[-1])
        nums = (int)(math.log(dim) / math.log(2.0)) - 2
        nums = min(nums, 3)
        return nums

    def _gauss2D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False):
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
        x = self.f_forward(x)
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
        ll = []
        ll.append(image.clone()[0].permute(2, 1, 0))
        output = sift_layer.forward(image)
        for num in range(sift_layer.nums):
            for j in range(sift_layer.Scale + 2):
                ll.append(sift_layer.octaves[num + 1].ImageLevel[j].level[0].permute(2, 1, 0))
                ll.append(sift_layer.DOGoctaves[num + 1].ImageLevel[j].level[0].permute(2, 1, 0))
        vis_img(ll, "image","gauss1", "chafen1", "gauss2", "chafen2", "gauss3", "chafen3", "gauss4", "chafen4", "gauss5",
                "chafen5", "gauss6", "chafen6", "gauss7", "chafen7", "gauss8", "chafen8", "gauss9", "chafen9",
                "gauss10", "chafen10", "gauss11", "chafen11", "gauss12", "chafen12")

    pass
