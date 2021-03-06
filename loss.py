import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        fusion_pred = pred
        fusion_pred = F.interpolate(fusion_pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        fusion_loss = self.fusion_loss(fusion_pred, target)
        return fusion_loss


# class GANLoss(nn.Module):
#     def __init__(self, cfg):
#         super(GANLoss, self).__init__()
#         self.threshold = cfg.GAN.THRESHOLD
#         self.batch_size = cfg.DATASET.BATCHSIZE
#         self.weight = cfg.LOSS.LOSS_WEIGHT
#         self.discriminator_loss = nn.BCELoss()
#
#     def forward(self, score_fake, score_real=None):
#         real = torch.FloatTensor(self.batch_size, 1).fill_(1).cuda()
#         soft_real = torch.FloatTensor(self.batch_size, 1).fill_(random.uniform(1-self.threshold, 1)).cuda()
#         fake = torch.FloatTensor(self.batch_size, 1).fill_(random.uniform(0, 0+self.threshold)).cuda()
#
#         if score_real is None:
#             loss = self.discriminator_loss(score_fake, real)
#         else:
#             loss_fake = self.discriminator_loss(score_fake, fake)
#             loss_real = self.discriminator_loss(score_real, soft_real)
#             loss = (loss_fake + loss_real) / 2
#         return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


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

    elif cfg.LOSS.TYPE == 'Contrastive':
        criterion = #TODO:ADD LOSS
    else:
        raise Exception("Criterion no found.")
    return criterion