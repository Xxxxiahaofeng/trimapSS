import math
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight_y = nn.Parameter(torch.zeros(in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        # TODO: This part will cost lot of gpu space, how to reduce it.
        # cause there is only 0 and 1,use a all 1 conv,
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.reshape((-1, self.in_channels, self.kernel_size**2, x.size(-1))) # (B, Cin, kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight_y.reshape((self.in_channels, self.kernel_size**2)) # (Cin, kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cin, kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # (B, Cin, kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x # (B, Cin, kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cin, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.in_channels, L_sqrt, L_sqrt)  # (B, Cin, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, kernel_size, soft_max, beta, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, in_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, kernel_size, soft_max, beta, 'erosion2d')


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


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
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
                   ]]).astype(np.float)
    y = torch.from_numpy(x)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    y = y.float()
    e = Erosion2d(2, 3, soft_max=False)
    d = Dilation2d(2, 3, soft_max=False)
    y1 = d(y)
    y2 = e(y)
    y = y1 - y2
    z = y.detach().numpy()
    print(z)

