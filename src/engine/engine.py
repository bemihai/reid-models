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
from utils.torchutils import save_checkpoint, load_checkpoint


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
            test_loss, test_metric = self._test_epoch()
            self._log_epoch('Testing', 1, max_epoch, test_loss, test_metric)
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        print('Start training')

        for epoch in range(start_epoch, max_epoch):
            train_loss, train_metric = self._train_epoch()
            self._log_epoch('Training', epoch+1, max_epoch, train_loss, train_metric)

            if (epoch + 1) >= start_eval \
                    and eval_freq > 0 \
                    and (epoch + 1) % eval_freq == 0 \
                    and (epoch + 1) != max_epoch:
                test_loss, test_metric = self._test_epoch()
                self._log_epoch('Testing', epoch+1, max_epoch, test_loss, test_metric)
                self._save_checkpoint(epoch+1, save_dir)

        if max_epoch > 0:
            print('Last epoch')
            test_loss, test_metric = self._test_epoch()
            self._log_epoch('Testing', max_epoch, max_epoch, test_loss, test_metric)
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


    def _train_epoch(self):
        """
        Performs training for one epoch.
        """
        if self.eval_metric:
            self.eval_metric.reset()

        self.model.train()
        training_loss = 0
        num_batches = len(self.train_loader)

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
            # compute evaluation metric
            self.eval_metric(outputs, target)

        if self.scheduler is not None:
            self.scheduler.step()

        # TODO: use average meter from utils
        training_loss /= num_batches
        return training_loss, self.eval_metric

    def _test_epoch(self):
        """
        Performs training for one epoch.
        """
        self.model.eval()
        testing_loss = 0
        num_batches = len(self.test_loader)

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
                testing_loss += loss.item()
                # compute evaluation metric
                self.eval_metric(outputs, target)

        # TODO: use average meter from utils
        testing_loss /= num_batches
        return testing_loss, self.eval_metric

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
            'epoch': epoch,
            'metric': self.eval_metric,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            }
        save_checkpoint(state, save_dir, remove_module_from_keys)

    @staticmethod
    def _log_epoch(stage, epoch, max_epoch, loss, metric):
        message = 'Epoch: {}/{}. {} average loss: {:.3f}'.format(epoch, max_epoch, stage, loss)
        message += '\t{}: {:.1f} %'.format(metric.name(), metric.value())
        print(message)



