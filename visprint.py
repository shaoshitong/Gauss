import cv2 as cv
import torch
import numpy as np
from matplotlib import pyplot as plt


def zys(n):
    value = []
    i = 2
    while i <= int(n / 2 + 1) and n != 1:
        print(i, n)
        if n % i == 0:
            value.append(i)
            n = n // i
            i -= 1
        i += 1
    value.append(1)
    return value


def vis_img(img):
    if img == None:
        pass
    output = img
    l = zys(len(output))
    l1 = 1
    l2 = 1
    tag = 0
    print(l)
    for i in l:
        if tag:
            l1 *= i
            tag = 0
        else:
            l2 *= i
            tag = 1
    print(l1, l2)
    for i, out in enumerate(output):
        b = torch.min(out).item()
        c = 255 / (torch.max(out).item() - b)
        out = (out - b) * c
        out = out.detach().numpy().astype(np.uint8)
        plt.subplot(l1, l2, i + 1)
        plt.imshow(out, "viridis")
        plt.title('img' + str(i))
    plt.show()
