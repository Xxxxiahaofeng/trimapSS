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
from loss import build_criterion
from optimizer import build_optimizer
from models.build_model import build_model
from utils import AverageTracker, runningScore, gen_color_map, logger

vis = visdom.Visdom()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, cfg):
        train_dataset = my_dataset(cfg.DATASET.TRAIN, split='train', n_class=cfg.DATASET.N_CLASS)
        self.train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.DATASET.BATCHSIZE,
                                                num_workers=8, shuffle=True)

        valid_dataset = my_dataset(cfg.DATASET.VAL, split='val', n_class=cfg.DATASET.N_CLASS)
        self.valid_dataloader = data.DataLoader(valid_dataset, batch_size=cfg.DATASET.BATCHSIZE,
                                              num_workers=8, shuffle=False)

        self.model = build_model(cfg).cuda(device)
        self.criterion = build_criterion(cfg).cuda(device)
        self.optimizer, self.lr_scheduler = build_optimizer(self.model, cfg)

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
            self.model.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['model_state'])
            self.optimizer.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['optimizer'])
            self.lr_scheduler.load_state_dict(
                torch.load(
                os.path.join(cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(self.start_epoch)))['lr_scheduler'])
            log = "Using the {}th checkpoint".format(self.start_epoch)
            self.logger.info(log)

    def train(self):

        all_train_iter_loss = []
        all_val_epo_acc = []
        all_val_epo_recall = []
        num_batches = len(self.train_dataloader)

        for epoch_i in range(self.start_epoch+1, self.n_epoch):

            iter_loss = AverageTracker()
            train_loss = AverageTracker()
            data_time = AverageTracker()
            batch_time = AverageTracker()
            tic = time.time()
            # train
            self.model.train()
            for i, meta in enumerate(self.train_dataloader):

                data_time.update(time.time() - tic)

                self.optimizer.zero_grad()

                image, target = meta[0].cuda(device), meta[1].cuda(device)
                pred, feat = self.model(image, epoch_i)
                loss = self.criterion(pred, target, epoch_i)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                iter_loss.update(loss.item())
                train_loss.update(loss.item())
                batch_time.update(time.time() - tic)
                tic = time.time()

                log = '{}: Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.6f}, Loss: {:.6f}' \
                    .format(datetime.now(), epoch_i, i, num_batches,
                            batch_time.avg, data_time.avg,
                            self.optimizer.param_groups[0]['lr'],
                            loss.item())
                print(log)

                if i % 10 == 0 and iter_loss.avg <= 10:
                    all_train_iter_loss.append(iter_loss.avg)
                    iter_loss.reset()
                    vis.line(all_train_iter_loss, win='train iter loss', opts=dict(title='train iter loss'))

            # eval
            self.model.eval()
            for j, meta in enumerate(self.valid_dataloader):
                image, target = meta[0].cuda(device), meta[1].cuda(device)
                preds = self.model(image, epoch_i)
                preds = F.softmax(preds, dim=1)
                preds = np.argmax(preds.cpu().detach().numpy().copy(), axis=1)
                target = target.cpu().detach().numpy().copy()
                self.running_metrics.update(target, preds)

                if j == 0:
                    color_map1 = gen_color_map(preds[0, :]).astype(np.uint8)
                    color_map2 = gen_color_map(preds[1, :]).astype(np.uint8)
                    color_map = cv2.hconcat([color_map1, color_map2])
                    cv2.imwrite(os.path.join(self.val_outdir, '{}epoch*{}*{}.png'
                                             .format(epoch_i, meta[4][0],meta[4][1])),
                                color_map)

            score = self.running_metrics.get_scores()
            oa = score['Overall Acc: \t']
            acc = score['Class Acc: \t'][1]
            recall = score['Recall: \t'][1]
            iou = score['Class IoU: \t'][1]
            F1 = score['F1 Score: \t']
            miou = score['Mean IoU: \t']
            self.running_metrics.reset()

            all_val_epo_acc.append(oa)
            all_val_epo_recall.append(recall)
            vis.line(all_val_epo_acc, win='val epoch acc', opts=dict(title='val epoch acc'))

            log = '{}: Epoch Val: [{}], ACC: {:.2f}, Recall: {:.2f}, mIoU: {:.4f}' \
                .format(datetime.now(), epoch_i, oa, recall, miou)
            self.logger.info(log)

            state = {'epoch': epoch_i,
                     "acc": oa,
                     "recall": recall,
                     "iou": miou,
                     'model_state': self.model.state_dict(),
                     'optimizer':self.optimizer.state_dict(),
                     'lr_scheduler':self.lr_scheduler.state_dict()}
            save_path = os.path.join(self.cfg.OUTDIR, 'checkpoints', '{}epoch.pth'.format(epoch_i))
            torch.save(state, save_path)


def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/hrnet_ppm_focus.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--opts",default=None , help="Modify config options using the command-line", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)
