import os
import torch
import torch.nn as nn
from models.backbones import *
from .decoders.global_decoder import *
from .decoders.transition_decoder import *


class build_model(nn.Module):
    def __init__(self, cfg):
        super(build_model, self).__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(cfg.MODEL.BACKBONE, cfg.PRETRAINED)
        self.global_decoder = self.build_global_decoder(cfg.MODEL.GLOBAL_DECODER)
        if self.cfg.MODEL.FPN:
            self.FPN = FPN(in_channels_list=cfg.MODEL.FC_DIM,
                           out_channels=cfg.MODEL.FPN_OUT,)
        if cfg.MODEL.FOCUS_DECODER:
            self.focus_decoder = self.build_focus_decoder()

    def forward(self, image, epoch_i):
        features = self.backbone(image)
        if self.cfg.MODEL.FPN:
            features = self.FPN(features)

        global_pred = self.global_decoder(features, self.cfg.MODEL.SEGSIZE)
        if self.cfg.MODEL.FOCUS_DECODER and epoch_i >= self.cfg.TRAIN_FOCUS:
            fusion_pred = self.focus_decoder(features, global_pred)
            if self.training:
                return global_pred, fusion_pred
            return F.interpolate(fusion_pred, size=self.cfg.MODEL.SEGSIZE, mode='bilinear', align_corners=False)
        else:
            if self.training:
                return global_pred
            return F.interpolate(global_pred, size=self.cfg.MODEL.SEGSIZE, mode='bilinear', align_corners=False)

    def build_backbone(self, backbone, pretrained=None):
        if backbone == 'resnet18':
            backbone = resnet18()
        elif backbone == 'resnet34':
            backbone = resnet34()
        elif backbone == 'resnet50':
            backbone = resnet50()
        elif backbone == 'resnet101':
            backbone = resnet101()
        elif backbone == 'resnet152':
            backbone = resnet152()
        elif backbone == 'HRNet':
            backbone = HRNetV2(self.cfg.MODEL.N_CLASS)
        else:
            raise Exception("Backbone no found.")

        if os.path.isfile(pretrained):
            pre_weight = torch.load(pretrained)
            model_dict = backbone.state_dict()
            pretrained_dict = {}
            for k, v in pre_weight.items():
                if k[6:] in model_dict.keys():
                    pretrained_dict[k[6:]] = v
            model_dict.update(pretrained_dict)
            backbone.load_state_dict(model_dict)

        return backbone

    def build_global_decoder(self, global_decoder):
        if global_decoder == 'PPM':
            global_decoder = PPM(n_class=self.cfg.MODEL.N_CLASS,
                                 fc_dim=self.cfg.MODEL.PPM_IN,
                                 pool_scales=self.cfg.MODEL.POOL_SCALES,
                                 segSize=self.cfg.MODEL.SEGSIZE)
        else:
            raise Exception("Global Decoder No Found.")
        return global_decoder

    def build_focus_decoder(self):
        return transition_decoder(inchannel=self.cfg.MODEL.PPM_IN,
                                  n_class=self.cfg.MODEL.N_CLASS,
                                  k_ed=self.cfg.MODEL.EROSION_DILATION_KERNAL,)
