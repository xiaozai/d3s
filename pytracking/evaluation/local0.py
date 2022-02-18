from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.mobiface_path = ''
    settings.network_path = '/home/yan/Data2/d3s/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = '/home/yan/Data2/d3s/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot16_path = '/home/yan/Data4/Datasets/vot2016/sequences/'
    settings.vot18_path = '/home/yan/Data4/Datasets/vot2018/sequences/'
    settings.vot_path = ''
    settings.cdtb_path = '/home/yan/Data4/Datasets/CDTB/sequences'
    settings.depthtrack_path = '/home/yan/Data4/Datasets/DepthTrack/'
    settings.davis16_path = '/home/yan/Data4/Datasets/DAVIS16/'
    settings.davis17_path = '/home/yan/Data4/Datasets/DAVIS17/'
    return settings
