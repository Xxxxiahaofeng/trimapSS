import torch
import numpy as np

labels_ = torch.from_numpy(np.random.randint(0, 10, [10,1]))
mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float()
logits_mask = torch.ones_like(mask).scatter_(1,
                                             torch.arange(10).view(-1, 1),
                                             0)
print(mask)