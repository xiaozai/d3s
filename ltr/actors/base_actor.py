from pytracking import TensorDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import datetime

class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective, target_sz_objective=None, target_size=False, loss_weights=None):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.target_size = target_size # Song
        self.target_sz_objective = target_sz_objective
        self.loss_weights = loss_weights

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """

        if torch.cuda.device_count()>1:
            print("Let's use ", torch.cuda.device_count(), "GPUs")
            self.net = nn.DataParallel(self.net)

        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)
