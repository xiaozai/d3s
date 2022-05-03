from .depth_segm import DepthSegmST
from .depth_segm_st_DAL import DepthSegmST as DepthSegmSTDAL
from .depth_segm_DualDCF import DepthSegmST as DepthSegmDual
from .depth_segm_rgbd_dcf import DepthSegmST as DepthSegmRGBDDCF

def get_tracker_class(model="depthsegm_st"):
    if model == 'depth_segm_st':
        return DepthSegmST
    elif model == 'depth_segm_dual':
        return DepthSegmDual
    elif model == 'depth_segm_st_DAL':
        return DepthSegmSTDAL
    elif model == 'depth_segm_rgbd_dcf':
        return DepthSegmRGBDDCF
    else:
        print("No such model :", model)
