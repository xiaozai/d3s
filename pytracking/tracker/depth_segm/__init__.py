from .depth_segm import DepthSegmST
from .depth_segm_st_DAL import DepthSegmST as DepthSegmSTDAL
from .depth_segm_DualDCF import DepthSegmST as DepthSegmDual

def get_tracker_class(model="depthsegm_st"):
    if model == 'depth_segm_st':
        return DepthSegmST
    elif model == 'depth_segm_dual':
        return DepthSegmDual
    elif model == 'depth_segm_st_DAL':
        return DepthSegmSTDAL
    else:
        print("No such model :", model)
