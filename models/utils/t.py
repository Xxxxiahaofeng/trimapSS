import torch
import numpy as np

labels_ = torch.from_numpy(np.random.randint(0, 10, [10,1]))
print(labels_[0:10,:])
