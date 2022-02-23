from .depth_segm_lt import DepthSegmLT
from .depth_segm_st import DepthSegmST

def get_tracker_class(model="depthsegm_st"):
    if model == 'depth_segm_st':
        return DepthSegmST
    elif model == 'depth_segm_lt':
        return DepthSegmLT
    else:
        print("No such model :", model)
