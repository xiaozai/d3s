from .depth_segm import DepthSegmST
from .depth_segm_st_DAL import DepthSegmST as DepthSegmSTDAL
from .depth_segm_rgbd_dcf import DepthSegmST as DepthSegmRGBDDCF
from .depth_segm_rgbd_feat_dcf import DepthSegmST as DepthSegmRGBDFeatDCF
from .depth_segm_rgbd_feat_dcf_post import DepthSegmST as DepthSegmRGBDFeatDCFPost
from .depth_segm_depth_dcf_post import DepthSegmST as DepthSegmDepthDCFPost
from .depth_segm_united import DepthSegmST as DepthSegmUnited

def get_tracker_class(model="depthsegm_st"):
    if model == 'depth_segm_st':
        return DepthSegmST
    elif model == 'depth_segm_st_DAL':
        return DepthSegmSTDAL
    elif model == 'depth_segm_rgbd_dcf':
        return DepthSegmRGBDDCF
    elif model == 'depth_segm_rgbd_feat_dcf':
        return DepthSegmRGBDFeatDCF
    elif model == 'depth_segm_rgbd_feat_dcf_post':
        return DepthSegmRGBDFeatDCFPost
    elif model == 'depth_segm_depth_dcf_post':
        return DepthSegmDepthDCFPost
    elif model == 'depth_segm_united':
        return DepthSegmUnited
    else:
        print("No such model :", model)
