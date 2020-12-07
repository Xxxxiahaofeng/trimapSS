import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Morphology(nn.Module):
    def __init__(self, channels, kernel_size, type=None):
        super(Morphology, self).__init__()
        self.channels = channels
        if kernel_size % 2 == 0:
            raise Exception('Kernal size must be odd.')
        self.kernal_size = kernel_size
        self.type = type

        weight = np.zeros((channels, 1, kernel_size, kernel_size))
        for i in range(kernel_size//2+1):
            weight[:, :, i, kernel_size//2-i:kernel_size//2+i+1] = 1
            weight[:, :, kernel_size-1-i, kernel_size//2-i:kernel_size//2+i+1] = 1
        self.register_buffer('morphology_weight', torch.from_numpy(weight).float())

    def forward(self, x):
        x = F.conv2d(x, weight=self.morphology_weight, padding=self.kernal_size//2, stride=1, groups=self.channels)
        if self.type == 'erosion2d':
            x = torch.where(x < (self.kernal_size**2+1)/2, 0., 1.)
        elif self.type == 'dilation2d':
            x = torch.where(x > 0, 1., 0.)
        return x


class Dilation2d(Morphology):
    def __init__(self, channels, kernel_size=5):
        super(Dilation2d, self).__init__(channels, kernel_size, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, channels, kernel_size=5):
        super(Erosion2d, self).__init__(channels, kernel_size, 'erosion2d')


if __name__ == '__main__':
    # test
    x = np.array([[[[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0.4, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0.8, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]],

                   [[0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0.7, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0]]
                   ]]).astype(np.float)
    y = torch.from_numpy(x)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    y = y.float()
    e = Erosion2d(2, 3)
    d = Dilation2d(2, 3)
    y1 = d(y)
    y2 = e(y)
    y = y1 - y2
    z = y.detach().numpy()
    print(z)