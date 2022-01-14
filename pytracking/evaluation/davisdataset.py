import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import pandas
import cv2
import os

def DavisDataset16():
    return DavisDatasetClass(version='16', res='480p').get_sequence_list()

def DavisDataset17():
    return DavisDatasetClass(version='17', res='480p').get_sequence_list()

class DavisDatasetClass(BaseDataset):
    """CDTB dataset"""
    def __init__(self, version='16', res='480p'):
        super().__init__()
        if version == '16':
            self.base_path = self.env_settings.davis16_path
        else:
            self.base_path = self.env_settings.davis17_path

        self.resolution = res
        self.sequence_list = self._get_sequence_list()


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 5
        ext = 'jpg'
        start_frame = 0

        # anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        # try:
        #     ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        # except:
        #     ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        # end_frame = ground_truth_rect.shape[0]
        end_frame = len(os.listdir('{base_path}/JPEGImages/{resolution}/{sequence_path}/'.format(base_path=self.base_path,
                  sequence_path=sequence_path, resolution=self.resolution, nz=nz)))-1

        frames = ['{base_path}/JPEGImages/{resolution}/{sequence_path}/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                  sequence_path=sequence_path, resolution=self.resolution, frame=frame_num, nz=nz)
                  for frame_num in range(start_frame, end_frame+1)]

        # mask is H*W binary mask
        init_mask = '{base_path}/Annotations/{resolution}/{sequence_path}/{frame:0{nz}}.png'.format(base_path=self.base_path,
                  sequence_path=sequence_path, resolution=self.resolution, frame=start_frame, nz=nz, ext=ext)
        init_mask = cv2.imread(init_mask, -1)

        nonzero_idx = np.nonzero(init_mask)
        row, col = nonzero_idx[0], nonzero_idx[1]
        min_y, min_x, max_y, max_x = min(row), min(col), max(row), max(col)
        init_bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
        ground_truth_rect = np.asarray([init_bbox])
        return Sequence(sequence_name, frames, ground_truth_rect, init_mask=init_mask)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # list_path = '{}/list.txt'.format(self.base_path)
        # sequence_list = pandas.read_csv(list_path, header=None, squeeze=True).values.tolist()
        sequence_list = os.listdir('{base_path}/JPEGImages/{resolution}/'.format(base_path=self.base_path, resolution=self.resolution))
        return sequence_list
