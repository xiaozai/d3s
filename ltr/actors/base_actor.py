from pytracking import TensorDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective, target_sz_objective=None, target_size=False):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.target_size = target_size # Song
        self.target_sz_objective = target_sz_objective

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
            # rank = 0
            # world_size = torch.cuda.device_count()
            # print(1, rank, world_size)
            # self.setup(rank, world_size)
            # print(2)
            # self.net = DDP(self.net, device_ids=[0, 1], output_device=0)
            # print(3)
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

    def setup(self, rank, world_size):
        # world_size,  Number of processes participating in the job
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()
