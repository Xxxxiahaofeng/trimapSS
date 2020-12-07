import torch


def build_optimizer(model, cfg):
    if cfg.OPTIMIZER.TYPE == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.OPTIMIZER.LR,
                                    momentum=cfg.OPTIMIZER.BETA1,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
                                    nesterov=True)
    elif cfg.OPTIMIZER.TYPE == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.OPTIMIZER.LR,
                                     betas=(cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2),
                                     weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,)
    else:
        raise Exception("Optimizer no found.")

    max_iter = cfg.MAX_ITER
    if cfg.OPTIMIZER.LR_SCHEDULER == 'poly':
        lr_lambda = lambda iter: (1-iter/max_iter)**0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise Exception("LR_Scheduler no found.")

    return optimizer, lr_scheduler
