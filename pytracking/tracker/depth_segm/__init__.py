from .depth_segm import DepthSegmST
from .depth_segm_redetection import DepthSegmST as DepthSegmRedet
from .depth_segm_st_DAL import DepthSegmST as DepthSegmSTDAL
from .depth_segm_rgbd_dcf import DepthSegmST as DepthSegmRGBDDCF
from .depth_segm_rgbd_feat_dcf import DepthSegmST as DepthSegmRGBDFeatDCF
from .depth_segm_rgbd_feat_dcf_post import DepthSegmST as DepthSegmRGBDFeatDCFPost
from .depth_segm_depth_dcf_post import DepthSegmST as DepthSegmDepthDCFPost
from .depth_segm_united import DepthSegmST as DepthSegmUnited
from .depth_segm_rgbd_feat_dcf_redetection import DepthSegmST as DepthSegmRGBDFeatDCFRedet
from .depth_segm_rgbd_cat_dcf_redetection import DepthSegmST as DepthSegmRGBDDCFCatRedet
from .depth_segm_pos_coatten_rgbd_dcf_redet import DepthSegmST as DepthSegmPosCoAttenDCFRedet
from .depth_segm_rgbd_feat_dcf_post_redet import DepthSegmST as DepthSegmRGBDFeatDCFPostRedet
from .depth_segm_rgbd_feat_dcf_pre_redet import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet
from .depth_segm_rgbd_feat_dcf_pre_redet02 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet02
from .depth_segm_rgbd_feat_dcf_pre_redet03 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet03
from .depth_segm_rgbd_feat_dcf_pre_redet04 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet04
from .depth_segm_rgbd_feat_dcf_pre_redet05 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet05
from .depth_segm_rgbd_feat_dcf_pre_redet06 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet06
from .depth_segm_rgbd_feat_dcf_pre_redet07 import DepthSegmST as DepthSegmRGBDFeatDCFPreRedet07
from .depth_segm_rgbd_feat_dcf_pre_post_redet import DepthSegmST as DepthSegmRGBDFeatDCFPrePostRedet

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
    elif model == 'depth_segm_rgbd_feat_dcf_redet':
        return DepthSegmRGBDFeatDCFRedet
    elif model == 'depth_segm_rgbd_cat_dcf_redet':
        return DepthSegmRGBDDCFCatRedet
    elif model == 'depth_segm_redet':
        return DepthSegmRedet
    elif model == 'depth_segm_pos_coatten_rgbd_dcf_redet':
        return DepthSegmPosCoAttenDCFRedet
    elif model == 'depth_segm_rgbd_feat_dcf_post_redet':
        return DepthSegmRGBDFeatDCFPostRedet
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet':
        return DepthSegmRGBDFeatDCFPreRedet
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet02':
        return DepthSegmRGBDFeatDCFPreRedet02
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet03':
        return DepthSegmRGBDFeatDCFPreRedet03
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet04':
        return DepthSegmRGBDFeatDCFPreRedet04
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet05':
        return DepthSegmRGBDFeatDCFPreRedet05
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet06':
        return DepthSegmRGBDFeatDCFPreRedet06
    elif model == 'depth_segm_rgbd_feat_dcf_pre_redet07':
        return DepthSegmRGBDFeatDCFPreRedet07
    elif model == 'depth_segm_rgbd_feat_dcf_pre_post_redet':
        return DepthSegmRGBDFeatDCFPrePostRedet
    else:
        print("No such model :", model)
