from pytracking.evaluation.environment import env_settings
import cv2
import numpy as np

class BaseDataset:
    """Base class for all datasets."""
    def __init__(self):
        self.env_settings = env_settings()

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError


class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, ground_truth_rect, object_class=None, init_mask=None):
        self.name = name
        self.frames = frames
        self.ground_truth_rect = ground_truth_rect
        self.init_state = list(self.ground_truth_rect[0,:])
        self.object_class = object_class


        ''' Song : added some term for RGBD'''
        self.init_mask = init_mask
        self.max_depth = None # only for RGBD dataset

        if isinstance(self.frames[0], dict) and ('depth' in self.frames[0]):
            init_depth = cv2.imread(self.frames[0]['depth'], -1)
            init_bbox = self.ground_truth_rect[0]
            if len(init_bbox) == 4:
                xywh = [int(float(b)) for b in init_bbox]
                depth_crop = init_depth[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]]
                depth_crop = np.nan_to_num(depth_crop)
                self.max_depth = np.median(depth_crop[depth_crop>0]) * 1.5
            else:
                print('not implement for polygon groundtruth in RGBD datasets...')

class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())
