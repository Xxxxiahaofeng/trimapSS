import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.morphology import Dilation2d, Erosion2d
from models.utils.comm import convBNReLU, label2onehot, init_weight


class transition_decoder(nn.Module):
    def __init__(self, inchannel, n_class, k_ed=5, segSize=None, alpha=0.1):
        """
        1) use erosion and dilation get transition area of object
        2) use erosion get certain area of objects, and sum to get object feature
        3) compute feature distance between transition area pixels and object pixel
        4) (1-transition_area)*global_feature +
           alpha*transition_area*global_feature +
           (1-alpha)*transition_area*distance*object_feature
        :param inchannel: output feature channel size of backbone
     m n_class: number of classes
        :param k_ed: kernel size of erosion and dilation
        :param segSize: original image size
        :param alpha: weight of global feature in transition area
        """
        super(transition_decoder, self).__init__()
        self.segSize = segSize
        self.n_class = n_class
        self.alpha = alpha

        self.fusion_predictor = nn.Sequential(convBNReLU(inchannel, 256),
                                              nn.Upsample(scale_factor=2, mode='bilinear')
                                              ,convBNReLU(256, 128),
                                              nn.Conv2d(128, n_class, kernel_size=1))

        self.dilation = Dilation2d(n_class, kernel_size=k_ed)
        self.erosion = Erosion2d(n_class, kernel_size=k_ed)

        init_weight(self)

    def forward(self, features, global_pred):
        feature = features[0]
        # generate the transition area from dilate and erode global prediction
        # transition area don't need gradient, cause it just decide where i want to do focus predict
        global_pred = global_pred.detach()
        _, global_pred = torch.max(global_pred, dim=1)
        global_pred = label2onehot(global_pred, self.n_class)

        d = self.dilation(global_pred)
        e = self.erosion(global_pred)

        transition = (d - e).sum(dim=1).unsqueeze(dim=1)
        transition[transition >= 1] = 1
        transition_feature = feature * transition

        objs = self.get_obj_feat(feature, e)
        dist = self.feat_obj_distance(transition_feature, objs, p=2)

        fusion_feat = feature*(1-transition) + \
                      feature*transition*self.alpha + \
                      (torch.matmul(dist, objs).transpose(1, 2).view(feature.shape)*transition)*(1 - self.alpha)
        fusion_pred = self.fusion_predictor(fusion_feat)

        if not self.training:
            fusion_pred = F.softmax(fusion_pred, dim=1)
        return fusion_pred

    def get_obj_feat(self, feature, erosion):
        """
        :param feature: shape [N, feature_dim, H, W]
        :param erosion: shape [N, class_num, H, W], only 0 and 1 in this map
        :return: objs shape [N, class_num, feature_dim]
        """
        objs = torch.cat([(feature * (erosion[:, c, :, :].unsqueeze(dim=1)))
                         .flatten(start_dim=2)
                         .mean(dim=2)
                         .unsqueeze(dim=1)
                          for c in range(erosion.size(1))],
                         dim=1)
        return objs

    def feat_obj_distance(self, feat, objs, p=2):
        """
        compute similarity of pixels in transition area to object feature
        :param feat: shape [N, feature_dim, H, W]
        :param objs: shape [N, class_num, feature_dim]
        :param p: Norm of distance computation
        :return: feature distance between pixels and object feature, size [N, HW, class_num]
        """
        shp = feat.shape
        feat = feat.flatten(start_dim=2).transpose(1, 2)

        dist = 1 - torch.cdist(feat, objs, p=p).softmax(-1)
        # dist = dist.transpose(1, 2).view(shp)
        return dist

