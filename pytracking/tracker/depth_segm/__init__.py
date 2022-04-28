from .depth_segm_lt import DepthSegmLT
# from .depth_segm_st import DepthSegmST
from .depth_segm import DepthSegmST
# from .depth_segm_st_ori import DepthSegmST as DepthSegmST02
from .depth_segm_st_DAL import DepthSegmST as DepthSegmSTDAL

def get_tracker_class(model="depthsegm_st"):
    if model == 'depth_segm_st':
        return DepthSegmST
    # elif model == 'depth_segm_st_ori':
    #     return DepthSegmST02
    elif model == 'depth_segm_st_DAL':
        return DepthSegmSTDAL
    elif model == 'depth_segm_lt':
        return DepthSegmLT
    else:
        print("No such model :", model)
