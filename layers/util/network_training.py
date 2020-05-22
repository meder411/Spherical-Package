import torch
import torch.nn as nn

import numpy as np

import random
import time
import sys
import os
import os.path as osp
import shutil
import logging
from enum import Enum

from .distributed import dprint, get_rank


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


def make_repeatable(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


def xavier_init(m):
    '''Provides Xavier initialization for the network weights and
    normally distributes batch norm params'''
    classname = m.__class__.__name__
    if (classname.find('Conv2d') !=
            -1) or (classname.find('ConvTranspose2d') !=
                    -1) or (classname.find('MappedConvolution') != -1) or (
                        classname.find('MappedTransposedConvolution') != -1):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def save_checkpoint(state, is_best, filename):
    '''Saves a training checkpoints'''
    torch.save(state, filename)
    if is_best:
        basename = osp.basename(filename)  # File basename
        idx = filename.find(basename)  # Where path ends and basename begins
        # Copy the file to a different filename in the same directory
        shutil.copyfile(filename, osp.join(filename[:idx], 'model_best.pth'))


def load_partial_model(model, loaded_state_dict):
    '''Loaded a save model, even if the model is not a perfect match. This will run even if there is are layers from the current network missing in the saved model.
    However, layers without a perfect match will be ignored.'''
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in loaded_state_dict.items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def load_optimizer(optimizer, loaded_optimizer_dict, device):
    '''Loads the saved state of the optimizer and puts it back on the GPU if necessary.  Similar to loading the partial model, this will load only the optimization parameters that match the current parameterization.'''
    optimizer_dict = optimizer.state_dict()
    pretrained_dict = {
        k: v
        for k, v in loaded_optimizer_dict.items()
        if k in optimizer_dict and k != 'param_groups'
    }
    optimizer_dict.update(pretrained_dict)
    optimizer.load_state_dict(optimizer_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class OpMode(Enum):
    """
    Mode of operation for the network
    """
    DEBUG = 1
    STANDARD = 2
    PRINT_ONLY = 4


def setup_logger(name, op_mode, logfile='', distributed_rank=0):
    logger = logging.getLogger(name)

    # Set the logging level
    level = logging.INFO
    if op_mode == OpMode.DEBUG:
        level = logging.DEBUG
    logger.setLevel(level)

    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    logger.addHandler(ch)

    if logfile:
        fh = logging.FileHandler(logfile, 'w')
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger


class TrainingEngine(object):

    def __init__(self,
                 network,
                 name='',
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 num_epochs=20,
                 validation_freq=1,
                 checkpoint_freq=1,
                 visualization_freq=5,
                 display_samples=True,
                 evaluation_sample_freq=-1,
                 logfile=None,
                 checkpoint_root='',
                 sample_root='',
                 op_mode=OpMode.STANDARD,
                 distributed=False,
                 device=None,
                 count_epochs=False,
                 higher_is_better=True):

        # Name of this experiment
        self.name = name

        # Class instances
        self.network = network
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training options
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.visualization_freq = visualization_freq
        self.display_samples = display_samples
        self.evaluation_sample_freq = evaluation_sample_freq
        self.count_epochs = count_epochs

        # Output directories
        self.checkpoint_dir = osp.join(checkpoint_root, name)
        self.sample_dir = osp.join(sample_root, name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # CUDA info
        self.device = device
        self.cuda_available = torch.cuda.is_available()

        # Some trackers
        self.epoch = 0

        # To store whatever the "best" metric is so far
        self.best_metric_key = 'best_metric'
        self.best_metric = 0.0 if higher_is_better else float('inf')
        self.is_best = False

        # Batch timer
        self.batch_time_meter = AverageMeter()

        # Mode of operation
        self.op_mode = op_mode
        self.distributed = distributed
        self.count_epochs = count_epochs

        # Create a logger
        self.logger = setup_logger(name, op_mode, logfile, get_rank())

    def print(self, *args, **kwargs):
        dprint(*args, **kwargs)

    def wrap_function_call(self, fn, *args, **kwargs):
        if self.op_mode == OpMode.DEBUG:
            if self.cuda_available:
                torch.cuda.synchronize()
            s = time.time()
            outputs = fn(*args, **kwargs)
            if self.cuda_available:
                torch.cuda.synchronize()
            self.logger.log_debug('{} seconds for function <{}>'.format(
                time.time() - s, fn.__name__))
            return outputs
        else:
            return fn(*args, **kwargs)

    def forward_pass(self, inputs, train=True):
        '''
        Accepts the inputs to the network as a Python list
        Returns the network output
        '''
        return self.network(*inputs)

    def compute_loss(self, output, gt):
        '''
        Returns the total loss
        '''
        return self.criterion(output, gt)

    def backward_pass(self, loss):
        # Computes the backward pass and updates the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_one_epoch(self):

        # Put the model in train mode
        self.network.train()

        # Load data
        end = time.time()
        for batch_num, data in enumerate(self.train_dataloader):

            # Parse the data into inputs, ground truth, and other
            inputs, gt, other = self.wrap_function_call(self.parse_data, data)

            # Run a forward pass
            output = self.wrap_function_call(self.forward_pass, inputs)

            # Compute the loss(es)
            loss = self.wrap_function_call(self.compute_loss, output, gt)

            # Compute the training metrics (if desired)
            self.wrap_function_call(self.compute_training_metrics, output, gt)

            # Backpropagation of the total loss
            self.wrap_function_call(self.backward_pass, loss)

            # Update batch times
            self.batch_time_meter.update(time.time() - end)
            end = time.time()

            if self.count_epochs:
                # Determine current batch as a function of (epochs, batch_num)
                current_batch_num = batch_num
            else:
                # Determine current batch as a function of total iterations run
                current_batch_num = self.epoch * len(
                    self.train_dataloader) + batch_num

            if current_batch_num % self.visualization_freq == 0:
                # Visualize the loss
                if self.op_mode.value < OpMode.PRINT_ONLY.value:
                    self.visualize_loss(current_batch_num, loss)
                    self.visualize_training_metrics(current_batch_num)
                    if self.display_samples:
                        self.visualize_samples(inputs, gt, other, output)
                self.print_batch_report(current_batch_num, loss)

    def train(self, checkpoint_path=None, weights_only=False):
        self.logger.info('Starting training')

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            if weights_only:
                self.initialize_visualizations()
        else:
            # Initialize any training visualizations
            self.initialize_visualizations()

        # Train for specified number of epochs
        if self.validation_freq >= 0:
            self.logger.info('Testing validation code...')
            self.validate(short=True)
        for self.epoch in range(self.epoch, self.num_epochs):

            # Clear all eval metrics
            self.reset_eval_metrics()

            # For distributed sampling, set_epoch must be called each time
            # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py#L25
            if isinstance(self.train_dataloader.sampler,
                          torch.utils.data.DistributedSampler):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            # Run an epoch of training
            self.train_one_epoch()

            if (self.validation_freq >=
                    0) and (self.epoch % self.validation_freq == 0):
                self.validate()
                self.visualize_metrics()
            if (self.checkpoint_freq >=
                    0) and (self.epoch % self.checkpoint_freq == 0):
                self.save_checkpoint()

            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

    def validate(self, short=False):
        self.logger.info('Validating model....')

        # Put the model in eval mode
        self.network.eval()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                self.print(
                    'Validating batch {}/{}'.format(batch_num,
                                                    len(self.val_dataloader)),
                    end='\r')

                # Parse the data
                inputs, gt, other = self.wrap_function_call(
                    self.parse_data, data, False)

                # Run a forward pass
                output = self.wrap_function_call(self.forward_pass, inputs,
                                                 False)

                # Compute the evaluation metrics
                self.wrap_function_call(self.compute_eval_metrics, output, gt)

                # If trying to save intermediate outputs
                if self.evaluation_sample_freq >= 0:
                    # Save the intermediate outputs
                    if batch_num % self.evaluation_sample_freq == 0:
                        self.save_samples(inputs, gt, other, output)

                # End validation after 5 batches if short==True
                # This is called before training to check that the validation step isn't broken, rather than waiting a whole epoch of training to find that out
                if short and batch_num > 4:
                    break

            # Any final computations on the eval metrics in aggregate
            self.wrap_function_call(self.compute_final_metrics)

        # Print a report on the validation results
        self.logger.info(
            'Validation finished in {} seconds'.format(time.time() - s))
        self.print_evaluation_report()

    def evaluate(self, checkpoint_path):
        self.logger.info('Evaluating model....')

        # Put the model in eval mode
        self.network.eval()

        # Load the checkpoint to test
        self.load_checkpoint(checkpoint_path, True, eval_mode=True)

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.test_dataloader):
                self.print(
                    'Evaluating batch {}/{}'.format(batch_num,
                                                    len(self.test_dataloader)),
                    end='\r')

                # Parse the data
                inputs, gt, other = self.wrap_function_call(
                    self.parse_data, data, False)

                # Run a forward pass
                output = self.wrap_function_call(self.forward_pass, inputs,
                                                 False)

                # Compute the evaluation metrics
                self.wrap_function_call(self.compute_eval_metrics, output, gt)

                # If trying to save intermediate outputs
                if self.evaluation_sample_freq >= 0:
                    # Save the intermediate outputs
                    if batch_num % self.evaluation_sample_freq == 0:
                        self.save_samples(inputs, gt, other, output)

            # Any final computations on the eval metrics in aggregate
            self.wrap_function_call(self.compute_final_metrics)

        # Print a report on the validation results
        self.logger.info(
            'Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_evaluation_report()

    def parse_data(self, data, train=True):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''
        raise NotImplementedError('Must implement the `parse_data` method')

    def compute_training_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the training of the model
        '''
        pass

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        pass

    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        pass

    def compute_final_metrics(self):
        """
        Called after the eval loop to perform any final computations (like aggregating metrics)
        """
        pass

    def initialize_visualizations(self):
        '''
        Initializes visualizations
        '''
        pass

    def visualize_loss(self, batch_num, loss):
        '''
        Updates the loss visualization
        '''
        pass

    def visualize_training_metrics(self, batch_num):
        '''
        Updates training metrics visualization
        '''
        pass

    def visualize_samples(self, inputs, gt, other, output):
        '''
        Updates the output samples visualization
        '''
        pass

    def visualize_metrics(self):
        '''
        Updates the metrics visualization
        '''
        pass

    def print_batch_report(self, batch_num, loss):
        '''
        Prints a report of the current batch
        '''
        pass

    def print_evaluation_report(self):
        '''
        Prints a report of the validation results
        '''
        pass

    def load_checkpoint(self,
                        checkpoint_path=None,
                        weights_only=False,
                        eval_mode=False):
        '''
        Initializes network with pretrained parameters
        '''
        if checkpoint_path is not None:
            self.logger.info(
                'Loading checkpoint \'{}\''.format(checkpoint_path))
            map_location = None
            if self.distributed:
                map_location = torch.device(get_rank())
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)

            # If we want to continue training where we left off, load entire training state
            if not weights_only:
                self.epoch = checkpoint['epoch'] - 1  # -1 to make it 0-indexed
                self.best_metric = checkpoint[self.best_metric_key]
                self.loss.from_dict(checkpoint['loss_meter'])
            else:
                # In evaluation mode, load the epoch for logging purposes
                if eval_mode:
                    self.epoch = checkpoint['epoch'] - 1  # -1 to make it 0-indexed
                self.logger.info('Loading weights only')

            # Load the optimizer and model state
            if self.optimizer is not None:
                if checkpoint['optimizer'] is not None:
                    load_optimizer(self.optimizer, checkpoint['optimizer'],
                                   self.device)
                else:
                    self.logger.warn('No optimization state in checkpoint!')
            load_partial_model(self.network, checkpoint['state_dict'])

            self.logger.info('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            self.logger.warn('No checkpoint found!')

    def save_checkpoint(self):
        '''
        Saves the model state
        '''
        if get_rank() != 0:
            return

        # Save latest checkpoint (constantly overwriting itself)
        checkpoint_path = osp.join(self.checkpoint_dir,
                                   'checkpoint_latest.pth')

        # Actually saves the latest checkpoint and also updating the file holding the best one
        save_checkpoint(
            {
                'epoch': self.epoch + 1,  # +1 to make it 1-indexed for humans
                'experiment': self.name,
                'state_dict': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_meter': self.loss.to_dict(),
                self.best_metric_key: self.best_metric
            },
            self.is_best,
            filename=checkpoint_path)

        # Copies the latest checkpoint to another file stored for each epoch
        history_path = osp.join(self.checkpoint_dir,
                                'checkpoint_{:03d}.pth'.format(self.epoch + 1))
        shutil.copyfile(checkpoint_path, history_path)
        self.logger.info('Checkpoint saved')

    def save_samples(self, inputs, gt, other, outputs):
        '''
        Saves samples of the network inputs and outputs
        '''
        pass
