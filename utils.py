import os
import torch
import logging
import numpy as np


class runningScore(object):

    def __init__(self, n_classes=11):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / (hist.sum() + 1e-6)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-6)
        recall = np.diag(hist) / hist.sum(axis=0)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-6)
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))
        F1 = 2 * acc_cls[1] * recall[1] / (acc_cls[1] + recall[1] + 1e-6)
        return {'Overall Acc: \t': acc,
                'Class Acc: \t': acc_cls,
                'Recall: \t': recall,
                'Class IoU: \t': cls_iu,
                'F1 Score: \t': F1,
                'Mean IoU: \t': mean_iu}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def label2onehot(label, n_class):
    label = label.unsqueeze(dim=1)
    onehot = torch.zeros((label.size()[0], n_class, label.size()[2], label.size()[3])).cuda().scatter_(1, label, 1)
    return onehot


def noise_onehot_label(label, n_class, threshold):
    label = label.unsqueeze(dim=1)
    onehot = torch.zeros((label.size()[0], n_class, label.size()[2], label.size()[3])).cuda().scatter_(1, label, 1)
    size = onehot.size()
    noise = threshold*torch.rand(size).cuda()
    noise = torch.where(onehot == 0, noise, -noise)
    noise_onehot = onehot + noise
    return noise_onehot


COLOR_TABLE = [
[0,   0,   0],  # 0
[128,128,128],  # 1
[128,0,  255],  # 2
[192,192,128],  # 3
[128,64, 128],  # 4
[60, 40, 222],  # 5
[128,128,0  ],  # 6
[192,128,128],  # 7
[64, 64, 128],  # 8
[64, 0,  128],  # 9
[64, 64, 0, ]]  # 10


CLR_CAMVID =[[64 ,128,64 ],[192,0  ,128],[0  ,128,192],[0  ,128,64 ],
             [128,0  ,0  ],[64 ,0  ,128],[64 ,0  ,192],[192,128,64 ],
             [192,192,128],[64 ,64 ,128],[128,0  ,192],[192,0  ,64 ],
             [128,128,64 ],[192,0  ,192],[128,64 ,64 ],[64 ,192,128],
             [64 ,64 ,0  ],[128,64 ,128],[128,128,192],[0  ,0  ,192],
             [192,128,128],[128,128,128],[64 ,128,192],[0  ,0  ,64 ],
             [0  ,64 ,64 ],[192,64 ,128],[128,128,0  ],[192,128,192],
             [64 ,0  ,64 ],[192,192,0  ],[0  ,0  ,0  ],[64 ,192,0  ]]


def gen_color_map(label):
    color_map = np.zeros((label.shape[0], label.shape[1], 3))
    for id in range(len(COLOR_TABLE)):
        color_map[label==id,0] = COLOR_TABLE[id][2]
        color_map[label==id,1] = COLOR_TABLE[id][1]
        color_map[label==id,2] = COLOR_TABLE[id][0]
    return color_map


LEVEL = {'DEBUG':logging.DEBUG,
         'INFO':logging.INFO,
         'WARNING':logging.WARNING,
         'ERROR':logging.ERROR,
         'CRITICAL':logging.CRITICAL}


def logger(path, name, level='INFO', if_cover=True):
    logger = logging.getLogger(name=name)
    logger.setLevel(LEVEL[level])
    if if_cover:
        file = logging.FileHandler(filename=os.path.join(path, 'log.txt'), mode='a')
    else:
        file = logging.FileHandler(filename=os.path.join(path, 'log.txt'))
    file.setLevel(LEVEL[level])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(LEVEL[level])

    logger.addHandler(file)
    logger.addHandler(console)

    return logger

