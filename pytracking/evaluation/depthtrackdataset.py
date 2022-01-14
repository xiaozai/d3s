from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import glob
import numpy as np
import os
import os.path as osp
from collections import OrderedDict
import pandas as pd

def DepthTrackDatasetTest():
    return DepthTrackDataset('test').get_sequence_list()
def DepthTrackDatasetTrain():
    return DepthTrackDataset('train').get_sequence_list()


class DepthTrackDataset(BaseDataset):
    """ DepthTrack RGBD Dataset
    """
    def __init__(self, split):
        """
        args:
            split - Split to use. Can be i) 'train': official training set, ii) 'test': official test set.
        """
        super().__init__()
        self.base_path = osp.join(self.env_settings.depthtrack_path, split)
        self.sequence_list = self._get_sequence_list()
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        # ext = 'jpg' # RGB is jpg, Depth is png
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = [{'color':'{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz),
                  'depth':'{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)}
                  for frame_num in range(start_frame, end_frame+1)]

        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_path = '{}/list.txt'.format(self.base_path)
        sequence_list = pandas.read_csv(list_path, header=None, squeeze=True).values.tolist()

        return sequence_list
