from .depth_segm import DepthSegm
from .depth_segm_st import DepthSegm_ST

def get_tracker_class(longterm=True):
    return DepthSegm if longterm else DepthSegm_ST
