import os
import cv2
import time
import torch
import visdom
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from datetime import datetime

from config import cfg
from dataset import my_dataset
from loss import build_criterion, GANLoss
from models.build_model import build_model
from models.GAN.discriminator import Discriminator
from utils import AverageTracker, runningScore, gen_color_map, logger, noise_onehot_label

vis = visdom.Visdom()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, cfg):
        train_dataset = my_dataset(cfg.DATASET.TRAIN, split='train', n_class=cfg.DATASET.N_CLASS)
        self.train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.DATASET.BATCHSIZE,
                                                num_workers=8, shuffle=True, drop_last=True)

        valid_dataset = my_dataset(cfg.DATASET.VAL, split='val', n_class=cfg.DATASET.N_CLASS)
        self.valid_dataloader = data.DataLoader(valid_dataset, batch_size=cfg.DATASET.BATCHSIZE,
                                              num_workers=8, shuffle=True, drop_last=True)

        self.generator = build_model(cfg).cuda(device)
        self.discriminator = Discriminator(cfg).cuda(device)

        self.criterion_G = build_criterion(cfg).cuda(device)
        self.criterion_D = GANLoss(cfg).cuda(device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=cfg.GAN.G_LR,
                                            betas=(cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2),
                                            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=cfg.GAN.D_LR,
                                            betas=(cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2),
                                            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        self.ckpt_outdir = os.path.join(cfg.OUTDIR, 'checkpoints')
        if not os.path.isdir(self.ckpt_outdir):
            os.mkdir(self.ckpt_outdir)
        self.val_outdir = os.path.join(cfg.OUTDIR, 'val')
        if not os.path.isdir(self.val_outdir):
            os.mkdir(self.val_outdir)
        self.start_epoch = cfg.RESUME
        self.max_iter = cfg.MAX_ITER
        self.n_epoch = cfg.N_EPOCH
        self.cfg = cfg

        self.logger = logger(cfg.OUTDIR, name='train')
        self.running_metrics = runningScore(n_classes=cfg.DATASET.N_CLASS)

        if self.start_epoch >= 0:
            self.generator.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['model_G'])
            self.discriminator.load_state_dict(
                torch.load(
                    os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['model_D'])
            self.optimizer_G.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['optimizer_G'])
            self.optimizer_D.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['optimizer_D'])

            log = "Using the {}th checkpoint".format(self.start_epoch)
            self.logger.info(log)

    def train(self):
        all_train_iter_total_loss = []
        all_train_iter_ss_loss = []
        all_train_iter_pd_loss = []
        all_train_iter_td_loss = []
        all_val_epo_iou = []
        all_val_epo_acc = []
        num_batches = len(self.train_dataloader)

        for epoch_i in range(self.start_epoch+1, self.n_epoch):
            iter_total_loss = AverageTracker()
            iter_ss_loss = AverageTracker()
            iter_pd_loss = AverageTracker()
            iter_td_loss = AverageTracker()
            batch_time = AverageTracker()
            tic = time.time()

            # train
            self.generator.train()
            for i, meta in enumerate(self.train_dataloader):

                image, target = meta[0].cuda(device), meta[1].cuda(device)

                # ---------------
                # Train Generator
                # ---------------
                self.optimizer_G.zero_grad()

                pred = self.generator(image, epoch_i)
                score_fake = self.discriminator(pred)

                semantic_loss = self.criterion_G(pred, target, epoch_i)
                gan_loss_4g = self.criterion_D(score_fake=score_fake)
                generator_loss = self.cfg.LOSS.LOSS_WEIGHT[0]*semantic_loss + self.cfg.LOSS.LOSS_WEIGHT[1]*gan_loss_4g

                generator_loss.backward()
                self.optimizer_G.step()

                # -------------------
                # Train Discriminator
                # -------------------
                self.optimizer_D.zero_grad()

                score_real = self.discriminator(noise_onehot_label(target, cfg.DATASET.N_CLASS, threshold=cfg.GAN.THRESHOLD))
                score_fake_d = self.discriminator(pred.detach())
                gan_loss_4d = self.criterion_D(score_fake=score_fake_d, score_real=score_real)

                gan_loss_4d.backward()
                self.optimizer_D.step()

                iter_total_loss.update(generator_loss.item())
                iter_ss_loss.update(semantic_loss.item())
                iter_pd_loss.update(gan_loss_4g.item())
                iter_td_loss.update(gan_loss_4d.item())
                batch_time.update(time.time() - tic)
                tic = time.time()

                log = '{}: Epoch: [{}][{}/{}], Time: {:.2f}, ' \
                      'Generator Total Loss: {:.6f}, Sementic Loss: {:.6f}, Pred Discriminator Loss: {:.6f}, Target Discriminator Loss: {:.6f}'.format(
                       datetime.now(), epoch_i, i, num_batches, batch_time.avg,
                       generator_loss.item(), semantic_loss.item(), gan_loss_4g.item(), gan_loss_4d.item())
                print(log)

                if i % 10 == 0:
                    all_train_iter_total_loss.append(iter_total_loss.avg)
                    iter_total_loss.reset()
                    all_train_iter_ss_loss.append(iter_ss_loss.avg)
                    iter_ss_loss.reset()
                    all_train_iter_pd_loss.append(iter_pd_loss.avg)
                    iter_pd_loss.reset()
                    all_train_iter_td_loss.append(iter_td_loss.avg)
                    iter_td_loss.reset()
                    vis.line(all_train_iter_total_loss, win='train iter generator total loss', opts=dict(title='train iter generator total loss'))
                    vis.line(all_train_iter_ss_loss, win='train iter generator semantic loss', opts=dict(title='train iter generator semantic loss'))
                    vis.line(all_train_iter_pd_loss, win='train iter pred discriminator loss', opts=dict(title='train iter pred discriminator loss'))
                    vis.line(all_train_iter_td_loss, win='train iter real discriminator loss', opts=dict(title='train iter real discriminator loss'))

            # eval
            self.generator.eval()
            for j, meta in enumerate(self.valid_dataloader):
                image, target = meta[0].cuda(device), meta[1].cuda(device)
                preds = self.generator(image, epoch_i)
                preds = F.softmax(preds, dim=1)
                preds = np.argmax(preds.cpu().detach().numpy().copy(), axis=1)
                target = target.cpu().detach().numpy().copy()
                self.running_metrics.update(target, preds)

                if j == 0:
                    color_map1 = gen_color_map(preds[0, :]).astype(np.uint8)
                    color_map2 = gen_color_map(preds[1, :]).astype(np.uint8)
                    color_map = cv2.hconcat([color_map1, color_map2])
                    cv2.imwrite(os.path.join(self.val_outdir, '{}epoch*{}*{}.png'
                                             .format(epoch_i, meta[4][0],meta[4][1])), color_map)

            score = self.running_metrics.get_scores()
            oa = score['Overall Acc: \t']
            acc = score['Class Acc: \t'][1]
            recall = score['Recall: \t'][1]
            iou = score['Class IoU: \t'][1]
            F1 = score['F1 Score: \t']
            miou = score['Mean IoU: \t']
            self.running_metrics.reset()

            all_val_epo_acc.append(oa)
            all_val_epo_iou.append(miou)
            vis.line(all_val_epo_acc, win='val epoch acc', opts=dict(title='val epoch acc'))
            vis.line(all_val_epo_iou, win='val epoch iou', opts=dict(title='val epoch iou'))

            log = '{}: Epoch Val: [{}], ACC: {:.2f}, Recall: {:.2f}, mIoU: {:.4f}' \
                .format(datetime.now(), epoch_i, oa, recall, miou)
            self.logger.info(log)

            state = {'epoch': epoch_i,
                     "acc": oa,
                     "recall": recall,
                     "iou": miou,
                     'model_G': self.generator.state_dict(),
                     'model_D': self.discriminator.state_dict(),
                     'optimizer_G':self.optimizer_G.state_dict(),
                     'optimizer_D':self.optimizer_D.state_dict()}
            save_path = os.path.join(self.cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(epoch_i))
            torch.save(state, save_path)


def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/resnet_GAN_focus.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--opts",default=None , help="Modify config options using the command-line", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)
