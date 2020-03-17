"""
Reference: torchreid
    - remove reid specific parts
    - replace torchreid datamanager with pytorch dataloader
"""

import time
import datetime
import torch

from torch.utils.tensorboard import SummaryWriter
from utils.utils import AverageMeter
from utils.torchutils import save_checkpoint, load_checkpoint, open_all_layers, open_specified_layers


class Engine(object):
    """
    Base Engine class.

    Args:
        train_loader: train data, an instance of ``torch.utils.data.DataLoader``
        test_loader: test data, an instance of ``torch.utils.data.DataLoader``
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        metric (Metric, optional): evaluation metric.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
            self,
            train_loader,
            test_loader,
            model,
            optimizer=None,
            scheduler=None,
            loss=None,
            metric=None,
            use_gpu=True
    ):
        # TODO: implement internal engine state
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.eval_metric = metric
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.writer = None

    # TODO: implement print freq and open/frozen layers
    def run(
            self,
            save_dir='runs',
            start_epoch=0,
            max_epoch=0,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
            start_eval=0,
            eval_freq=-1,
            test_only=False,
    ):
        """
        A unified pipeline for training and testing a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test dataset.
                Default is False.
        """

        if test_only:
            self._test_epoch(start_epoch, max_epoch)
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        print('Start training')

        for epoch in range(start_epoch, max_epoch):
            self._train_epoch(epoch, max_epoch, fixbase_epoch, open_layers, print_freq)

            if (epoch + 1) >= start_eval \
                    and eval_freq > 0 \
                    and (epoch + 1) % eval_freq == 0 \
                    and (epoch + 1) != max_epoch:
                self._test_epoch(epoch, max_epoch)
                self._save_checkpoint(epoch, save_dir)

        if max_epoch > 0:
            print('Last epoch')
            self._test_epoch(max_epoch, max_epoch)
            self._save_checkpoint(max_epoch, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed training time {}'.format(elapsed))

        if self.writer:
            self.writer.close()

    def resume_from_checkpoint(
            self,
            fpath,
            save_dir='runs',
            max_epoch=0,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
            start_eval=0,
            eval_freq=-1,
            test_only=False,
    ):
        r"""
        Resumes training from a checkpoint.

        Args:
            fpath (str): path to checkpoint.
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test dataset.
                Default is False.
        """
        print('Loading checkpoint from "{}"'.format(fpath))
        checkpoint = load_checkpoint(fpath)
        assert 'state_dict' in checkpoint.keys()
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model weights')
        if self.optimizer and 'optimizer' in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded optimizer')
        if self.scheduler and 'scheduler' in checkpoint.keys():
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print('Loaded scheduler')
        start_epoch = checkpoint['epoch']
        message = 'Resume training from epoch {}'.format(start_epoch)
        if 'metric' in checkpoint.keys():
            message += ' ({}: {:.1f}%)'.format(checkpoint['metric'].name(), checkpoint['metric'].value())
        print(message)
        self.run(save_dir, start_epoch, max_epoch, print_freq,
                 fixbase_epoch, open_layers, start_eval, eval_freq, test_only)


    def _train_epoch(self, epoch, max_epoch, fixbase_epoch, open_layers, print_freq):
        """
        Performs training for one epoch.
        """
        if self.eval_metric:
            self.eval_metric.reset()

        self.model.train()
        training_loss = 0
        num_batches = len(self.train_loader)

        losses = AverageMeter('Loss')
        metrics = AverageMeter('Accuracy')
        batch_time = AverageMeter('Batch time')

        if (epoch + 1) <= fixbase_epoch and open_layers:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # parse inputs
            data, target = self._parse_inputs(data, target)
            # extract features
            self.optimizer.zero_grad()
            outputs = self.model(*data)
            # compute loss and do backward step
            loss = self._compute_loss(outputs, target)
            training_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            losses.update(loss.item())
            # compute evaluation metric
            self.eval_metric(outputs, target)
            metrics.update(self.eval_metric.value())

            if (batch_idx + 1) % print_freq == 0:
                self._log_train_iteration(epoch, max_epoch, batch_idx, num_batches, batch_time, losses, metrics)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def _test_epoch(self, epoch, max_epoch):
        """
        Performs training for one epoch.
        """
        self.model.eval()

        losses = AverageMeter('Loss')
        metrics = AverageMeter('Accuracy')

        with torch.no_grad():
            # reset evaluation metric
            if self.eval_metric:
                self.eval_metric.reset()

            for batch_idx, (data, target) in enumerate(self.test_loader):
                # parse inputs
                data, target = self._parse_inputs(data, target)
                # extract features
                outputs = self.model(*data)
                # compute loss
                loss = self._compute_loss(outputs, target)
                losses.update(loss.item())
                # compute evaluation metric
                self.eval_metric(outputs, target)
                metrics.update(self.eval_metric.value())

            self._log_test_iteration(epoch, max_epoch, losses, metrics)

    def _parse_inputs(self, data, target):
        target = (target.to(self.device),) if len(target) > 0 else None
        if type(data) not in (tuple, list):
            data = (data,)
        data = tuple(d.to(self.device) for d in data)
        return data, target

    def _compute_loss(self, outputs, target):
        loss_inputs = outputs
        if target:
            loss_inputs += target
        loss_outputs = self.loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        return loss

    def _save_checkpoint(self, epoch, save_dir, remove_module_from_keys=False):
        state = {
            'model': 'osnet',
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'metric': self.eval_metric,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            }
        save_checkpoint(state, save_dir, remove_module_from_keys)

    @staticmethod
    def _log_train_iteration(epoch, max_epoch, batch_idx, num_batches, batch_time, losses, metrics):
        eta_sec = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        progress = 100*(batch_idx + 1)/num_batches
        print(
            'Epoch: [{0}/{1}][ {prog:.0f}% ] ' 
            'Avg Train Loss: {loss.avg:.3f} | '
            'Avg Train {metrics.name}: {metrics.avg:.2f}% | '
            'ET: {eta}'.format(epoch + 1, max_epoch, prog=progress, loss=losses, metrics=metrics, eta=eta_str)
        )

    @staticmethod
    def _log_test_iteration(epoch, max_epoch, losses, metrics):
        print(
            'Epoch: [{0}/{1}] ' 
            'Avg Test Loss: {loss.avg:.3f} | '
            'Avg Test {metrics.name}: {metrics.avg:.2f}% '
            .format(epoch + 1, max_epoch, loss=losses, metrics=metrics)
        )



