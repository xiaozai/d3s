import ltr
from ltr import load_network
import torch

path = '/home/sgn/Data1/yan/d3s/checkpoints/ltr/depth_segm/depth_only_segm/DepthSegmNet_ep0040.pth.tar'

checkpoint_dict = torch.load(path)
torch.save(checkpoint_dict, path[:-8]+'_02'+'.pth.tar', _use_new_zipfile_serialization=False) # Song
