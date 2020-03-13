""" CIFAR-10 classification """

import os
import contextlib
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn

from optim.optimizer import build_optimizer
from optim.scheduler import build_lr_scheduler
from losses.loss import CrossEntropyLoss

from config.config import get_cfg
from models.networks import ClassificationNet
from engine.engine import Engine
from models.osnet import OSBlock, OSNet
from metrics.accuracy import AccumulatedAccuracy

train_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.247, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.247, 0.2435, 0.2616)),
])

# disable messages printed by DataLoader
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    cifar_training = datasets.CIFAR10('../data/', train=True, download=True, transform=train_transform)
    cifar_testing = datasets.CIFAR10('../data/', train=False, download=True, transform=train_transform)

def main():
    cfg = get_cfg()                                          # default config
    cfg.use_gpu = torch.cuda.is_available()                  # device: cuda
    cfg.model.feature_dim = 128                              # nr of features to be extracted
    cfg.train.max_epoch = 50                                 # nr of epochs
    cfg.train.start_epoch = 0                                # starting epoch
    cfg.train.batch_size = 64                                # mini batch size
    cfg.train.optim = 'sgd'                                  # optimizer
    cfg.train.lr = 0.003                                     # learning rate
    cfg.train.weight_decay = 5e-4                            # weight decay (L2-regularization)
    cfg.train.lr_scheduler = 'single_step'                   # learning rate scheduler
    cfg.train.stepsize = [20]                                # stepsize to decay learning rate
    cfg.train.gamma = 0.1                                    # learning rate decay multiplier
    cfg.loss.name = 'softmax'                                # loss function name
    cfg.loss.softmax.label_smooth = True                     # label smoothing regularizer

    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_gpu else {}
    train_loader = DataLoader(cifar_training, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(cifar_testing, batch_size=cfg.train.batch_size, shuffle=False, drop_last=True, **kwargs)

    feature_extractor = OSNet(
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        feature_dim=cfg.model.feature_dim
    )

    model = ClassificationNet(
        feature_extractor,
        feature_dim=cfg.model.feature_dim,
        n_classes=10
    )

    if cfg.use_gpu:
        cudnn.benchmark = True
        model = nn.DataParallel(model).cuda()

    optimizer = build_optimizer(
        model,
        optim=cfg.train.optim,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    lr_scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler=cfg.train.lr_scheduler,
        stepsize=cfg.train.stepsize,
        gamma=cfg.train.gamma
    )

    loss = CrossEntropyLoss(
        num_classes=10,
        use_gpu=cfg.use_gpu,
        label_smooth=cfg.loss.softmax.label_smooth
    )

    engine = Engine(
        train_loader=train_loader,
        test_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss=loss,
        use_gpu=True,
        metric=AccumulatedAccuracy(),
    )

    # engine.run(
    #     start_epoch=0,
    #     max_epoch=cfg.train.max_epoch,
    #     start_eval=0,
    #     eval_freq=1,
    # )

    engine.resume_from_checkpoint(
        fpath='runs/osnet-epoch-7.pth.tar',
        max_epoch=cfg.train.max_epoch,
        start_eval=0,
        eval_freq=1,
    )


if __name__ == "__main__":
    main()


