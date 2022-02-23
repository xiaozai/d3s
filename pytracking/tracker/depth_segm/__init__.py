from .depth_segm import DepthSegmLT
from .depth_segm_st import DepthSegmST
# from .depth_init_segm import DepthInitSegm
# from .depth_only_segm import DepthOnlySegm
# from .depth_segm_attention import DepthSegmAttention

def get_tracker_class(model="depthsegm_st"):

    # if model == "depthsegm_st":
    #     return DepthSegm_ST
    # elif model == "depth_only_segm":
    #     return DepthOnlySegm
    # elif model == "depth_init_segm":
    #     return DepthInitSegm
    # elif model == "depth_segm_attention":
    #     return DepthSegmAttention
    # elif model == 'depth_segm_attention1':
    #     return DepthSegmAttention1
    # elif model == 'depth_segm_attention2':
    #     return DepthSegmAttention2
    # elif model == "depth_segm_lt":
    #     return DepthSegm
    if model == 'depth_segm_st':
        return DepthSegmST
    elif model == 'depth_segm_lt':
        return DepthSegmLT
    else:
        print("No such model :", model)
