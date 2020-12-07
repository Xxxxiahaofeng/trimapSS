import torch
import numpy as np
import torch.nn as nn


def convBNReLU(inchannel, outchannel, stride=1):
    return nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(outchannel),
                         nn.ReLU(inplace=True))


def convReLU(inchannel, outchannel, stride=1):
    return nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                         nn.ReLU(inplace=True))


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def label2onehot(label, n_class):
    label = label.unsqueeze(dim=1)
    onehot = torch.zeros((label.size()[0], n_class, label.size()[2], label.size()[3])).cuda().scatter_(1, label, 1)
    return onehot


if __name__ == '__main__':
    label = np.array([[[[1, 0 ,2],
                      [0, 0, 0],
                      [3, 0 ,1]]]])
    label = torch.from_numpy(label)
    onehot = label2onehot(label, 4)
    onehot = onehot.detach().numpy()
    print(onehot)