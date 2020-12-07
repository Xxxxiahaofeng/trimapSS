import os
import cv2
import torch
import numpy as np
from torch.utils import data

class my_dataset(data.Dataset):
    def __init__(self, datalist_dir, split='train', n_class=11):
        if split == 'train':
            with open(os.path.join(datalist_dir, 'train.txt'))as f:
                self.datalist = f.readlines()
        elif split == 'val':
            with open(os.path.join(datalist_dir, 'val.txt'))as f:
                self.datalist = f.readlines()
        self.mean = np.array([72.57055382, 90.68219696, 81.40952313])
        self.var = np.array([51.17250644, 53.22876833, 60.39464522])
        self.split = split
        self.num_class = n_class

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, item):
        imgdir, labeldir, _ = self.datalist[item].replace("\n", "").split(" ")
        labelname = labeldir[labeldir.rindex('/') + 1:]
        labelname = labelname[:labelname.index('.')]

        img = cv2.imread(imgdir, flags=cv2.IMREAD_COLOR)
        label = cv2.imread(labeldir, flags=cv2.IMREAD_GRAYSCALE)

        img = self.transform(img)
        label[label <= 0] = 0
        label[label >= self.num_class] = 10
        # print(labelname)
        label = label.astype(np.int)

        height = img.shape[1]
        width = img.shape[2]
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        if self.split == 'val':
            return img, label, height, width, labelname
        return img, label, height, width, labelname

    def transform(self, img):
        img = img.astype(float)
        img -= self.mean
        img /= self.var
        img = img.transpose(2, 0, 1)

        return img