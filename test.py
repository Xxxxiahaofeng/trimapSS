import os
import cv2
import torch
import argparse
import numpy as np
from config import cfg
from utils import gen_color_map
import torch.nn.functional as F
from models.build_model import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def patch_size(imagesize, patch_size, overlap):
    img_h, img_w = imagesize
    p_h = patch_size
    p_w = patch_size
    start_h, start_w = 0, 0
    start_hs = []
    start_ws = []

    while start_h < img_h:
        start_hs.append(start_h)
        start_h = int(start_h + (1 - overlap) * p_h - 1)
        if start_h+p_h >= img_h:
            start_h = img_h - p_h - 1
            start_hs.append(start_h)
            break

    while start_w < img_w:
        start_ws.append(start_w)
        start_w = int(start_w + (1 - overlap) * p_w - 1)
        if start_w+p_w >= img_w:
            start_w = img_w - p_w - 1
            start_ws.append(start_w)
            break

    return start_hs, start_ws


def transform(img):
    mean = np.array([72.57055382, 90.68219696, 81.40952313])
    var = np.array([51.17250644, 53.22876833, 60.39464522])

    img = img.astype(float)
    img -= mean
    img /= var
    img = img.transpose(2, 0, 1)

    return img


def main(cfg):
    # load image
    image = cv2.imread(cfg.TEST.TESTIMAGE, flags=cv2.IMREAD_COLOR)
    result = np.zeros_like(image)
    size_h, size_w = image.shape[:2]
    image = transform(image)
    image = image[np.newaxis, :, :, :]

    # init model
    model = build_model(cfg).cuda(device)
    ckpt_dir = cfg.TEST.CHECKPOINT
    print(torch.cuda.is_available())
    model_state_dict = torch.load(ckpt_dir)['model_state']
    model.load_state_dict(model_state_dict)
    model.cuda()

    start_hs, start_ws = patch_size([size_h, size_w], cfg.TEST.PATCHSIZE, cfg.TEST.OVERLAP)

    model.eval()
    for start_h in start_hs:
        for start_w in start_ws:
            img_patch = image[:, :, start_h:start_h+cfg.TEST.PATCHSIZE, start_w:start_w+cfg.TEST.PATCHSIZE]
            img_patch = torch.from_numpy(img_patch).float().cuda()

            pred = model(img_patch, 100000)
            pred = F.softmax(pred, dim=1)
            pred = np.argmax(pred.cpu().detach().numpy().copy(), axis=1)
            color_map = gen_color_map(pred[0])
            result[start_h:start_h+cfg.TEST.PATCHSIZE, start_w:start_w+cfg.TEST.PATCHSIZE, :] = color_map

    cv2.imwrite(cfg.TEST.OUTDIR, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/hrnet_ppm_focus.yaml", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--opts", default=None, help="Modify config options using the command-line",
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)