import numpy as np
import torch
import torch.nn as nn

input1 = torch.randn(1, 64, 1)
input2 = torch.randn(1, 11, 1)
dist = torch.cdist(input1, input2, p=2)
output = torch.matmul(dist, input2)
print(input1)
print(input2)
print(dist)