import torch
import torch.nn as nn
import torch.nn.functional as F


class MixLoss(nn.Module):
    def __init__(self, n_class, ignore_index=-1, loss_weight=(1,1,1), class_weight=None, focus_start=0):
        super(MixLoss, self).__init__()
        self.n_class = n_class
        self.fucus_start = focus_start
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.global_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)
        self.fusion_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)

    def forward(self, pred, target, epoch_i):
        if epoch_i >= self.fucus_start:
            global_pred, fusion_pred = pred
        else:
            global_pred = pred

        global_pred = F.interpolate(global_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        global_loss = self.global_loss(global_pred, target)
        loss = self.loss_weight[0] * global_loss

        if epoch_i >= self.fucus_start:
            fusion_pred = F.interpolate(fusion_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            fusion_loss = self.fusion_loss(fusion_pred, target)
            loss += self.loss_weight[1] * fusion_loss

        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, weight=None, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred, target, epoch_i):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class CELoss(nn.Module):
    def __init__(self, ignore_index=-1, class_weight=None):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.class_weight = class_weight

        self.fusion_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)

    def forward(self, pred, target, epoch_i):
        fusion_pred = pred[-1]

        fusion_pred = F.interpolate(fusion_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        fusion_loss = self.fusion_loss(fusion_pred, target)
        return fusion_loss


def build_criterion(cfg):
    if cfg.LOSS.TYPE == 'MixLoss':
        criterion = MixLoss(n_class=cfg.DATASET.N_CLASS,
                            ignore_index=cfg.LOSS.IGNORE_INDEX,
                            loss_weight=cfg.LOSS.LOSS_WEIGHT,
                            class_weight=cfg.LOSS.CLASS_WEIGHT,
                            focus_start = cfg.TRAIN_FOCUS)

    elif cfg.LOSS.TYPE == 'OHEMLoss':
        criterion = OhemCrossEntropy2d(ignore_index=cfg.LOSS.IGNORE_INDEX,
                                       thresh=cfg.LOSS.THRESHOLD,
                                       min_kept=cfg.LOSS.MIN_KEPT,
                                       weight=cfg.LOSS.CLASS_WEIGHT)

    elif cfg.LOSS.TYPE == 'CrossEntropy':
        criterion = CELoss(class_weight=cfg.LOSS.CLASS_WEIGHT,
                           ignore_index=cfg.LOSS.IGNORE_INDEX)
    else:
        raise Exception("Criterion no found.")
    return criterion