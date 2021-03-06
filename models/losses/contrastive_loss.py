import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelContrastLoss(nn.Module):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.CONTRAST.TEMPERATURE
        self.base_temperature = self.configer.CONTRAST.BASE_TEMPERATURE

        self.max_samples = self.configer.CONTRAST.MAX_SAMPLES
        self.max_views = self.configer.CONTRAST.MAX_VIEWS

    def _hard_anchor_sampling(self, X, y_hat, y): # feature [B,HxW,F], label [B,HxW], predict [B,CxHxW]
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            # get labels in image_ii
            this_classes = torch.unique(this_y)
            # abandon invalid labels
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            # assert class to use has least num of pixels > max_views
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # classes record valid labels in image_ii
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None
        # max_samples: total samples in a batch, n_view: number of pixels to sample in one class of one image
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                # hard pixels: gt_id != pred_id, easy pixels: gt_id == pred_id
                # nonzero return index of nonzero elements
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                # how to assign hard and easy samples
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception
                # return a shuffled list from 0 to num_hard-1
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                # X_:[BxN_class_ii,n_view,F], y_:[BxN_class_ii]
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        # mask: [BxN_class_ii, BxN_class_ii], record which samples have same label
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)    # contrast_feature: [B x N_class_ii x n_view, F]

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # anchor_dot_contrast:[B x N_class_ii x n_view, B x N_class_ii x n_view]
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # [B x N_class_ii x n_view,B x N_class_ii x n_view], NOTE keepdim = True
        logits = anchor_dot_contrast - logits_max.detach()

        # repeat mask n_view time, now mask:[B x N_class_ii x n_view, B x N_class_ii x n_view]
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask        # make a all one mask, while diagonal is 0
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        # diagonal of mask become 0
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # NOTE log function is separated
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)   # [B,HxW]
        predict = predict.contiguous().view(batch_size, -1) # [B,CxHxW]
        feats = feats.permute(0, 2, 3, 1) # [B,H,W,F]
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) # [B,HxW,F]
        # feats_:[BxNum_class_ii,n_view,F], labels_:[BxNum_class_ii]
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss

class RegionContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(RegionContrastiveLoss, self).__init__()
        self.config = config
        self.queue_size = self.config.CONTRAST.QUEUE_SIZE
        self.temperature = self.config.CONTRAST.TEMPERATURE

        self.register_buffer("queue", torch.randn(self.config.CONTRAST.QUQUE_SIZE, self.config.MODEL.FC_DIM[-1]))
        self.register_buffer("y_", torch.zeros(self.config.CONTRAST.QUEUE_SIZE, 1) - 1)
        self.ptr = 0

    def enqueue(self, feature, y_):
        """
        :param feature: size [B x ClassNum_ii, F]
        :param y_: [B x ClassNum_ii, 1]
        """
        batch_size = feature.shape[0]
        remain = self.queue_size - self.ptr

        if remain >= batch_size:
            self.queue[self.ptr:self.ptr+batch_size, :] = feature
            self.y_[self.ptr:self.ptr+batch_size, :] = y_
            self.ptr = (self.ptr + batch_size) % self.queue_size
        elif 0 < remain < batch_size:
            self.queue[self.ptr:, :] = feature[:remain, :]
            self.queue[0:batch_size-remain, :] = feature[remain:, :]
            self.y_[self.ptr:, :] = y_[:remain, :]
            self.y_[0:batch_size-remain, :] = y_[remain:, :]
            self.ptr = batch_size-remain

    def forward(self, feats, labels):
        """
        :param feats: size [B,F,H,W]
        :param labels: size [B,H,W]
        """
        #TODO:SUM FEATURES IN SAME GT LABEL
        
