"""
Reference: torchreid
    - remove reid specific parts
    - replace torchreid datamanager with pytorch dataloader
"""

import time
from abc import ABC, abstractmethod
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

# from torchreid.utils import save_checkpoint


class Engine(ABC):
    """
    A generic base Engine class.

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
                # self._save_checkpoint(epoch, test_metric, save_dir)

        if max_epoch > 0:
            print('Last epoch')
            test_loss, test_metric = self._test_epoch()
            self._log_epoch('Testing', max_epoch, max_epoch, test_loss, test_metric)
            # self._save_checkpoint(epoch, test_metric, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed training time {}'.format(elapsed))

        if self.writer:
            self.writer.close()

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
            # parse data for training
            target = (target.to(self.device),) if len(target) > 0 else None
            if type(data) not in (tuple, list):
                data = (data,)
            data = tuple(d.to(self.device) for d in data)
            # extract features
            self.optimizer.zero_grad()
            outputs = self.model(*data)
            # parse loss inputs
            loss_inputs = outputs
            if target:
                loss_inputs += target
            # compute loss
            loss_outputs = self.loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            training_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            # compute evaluation metric
            self.eval_metric(outputs, target)

        if self.scheduler is not None:
            self.scheduler.step()

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
                # parse data for testing
                target = (target.to(self.device),) if len(target) > 0 else None
                if type(data) not in (tuple, list):
                    data = (data,)
                data = tuple(d.to(self.device) for d in data)
                # extract features
                outputs = self.model(*data)
                # parse features
                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                # parse loss inputs
                loss_inputs = outputs
                if target:
                    loss_inputs += target
                # compute loss
                loss_outputs = self.loss_fn(*loss_inputs)
                # parse loss outputs
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                testing_loss += loss.item()
                # compute evaluation metric
                self.eval_metric(outputs, target)

        testing_loss /= num_batches
        return testing_loss, self.eval_metric

    def _extract_features(self, x):
        self.model.eval()
        return self.model(x)

    @staticmethod
    def _log_epoch(stage, epoch, max_epoch, loss, metric):
        message = 'Epoch: {}/{}. {} average loss: {:.3f}'.format(epoch, max_epoch, stage, loss)
        message += '\t{}: {:.1f} %'.format(metric.name(), metric.value())
        print(message)

    @staticmethod
    def _parse_data_for_training(data):
        pass

    @staticmethod
    def _parse_data_for_testing(data):
        pass

    @staticmethod
    def _parse_features(data):
        pass

    def _compute_loss(self):
        pass

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best
        )
