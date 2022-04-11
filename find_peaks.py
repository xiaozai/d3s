import os
import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':

    data_path = '/home/yan/Data2/VOT/vot-workspace-CDTB/sequences/'
    seq = 'box_darkroom_noocc_6'

    depth_path = data_path + seq + '/' + 'depth/'

    gt_path = data_path + seq + '/' 'groundtruth.txt'

    with open(gt_path, 'r') as fp:
        boxes = fp.readlines()
    boxes = [box.strip() for box in boxes]

    frame_id = 1
    depth = cv2.imread(depth_path+'%08d.png'%frame_id, -1)
    box = boxes[frame_id-1]
    box = [int(float(b)) for b in box.split(',')]
    num_pixels = box[2]*box[3]
    print('num pixels: ', num_pixels)

    depth_crop = np.asarray(depth[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
    print('max depth:', np.max(depth_crop), '  min depth: ', np.min(depth_crop))

    depth_crop = medfilt(depth_crop, 15)
    # hist values, bin edges
    depth_hist = np.histogram(depth_crop.flatten(), bins=20)
    # print(depth_hist)


    peaks = find_peaks(depth_hist[0], height=num_pixels/10)[0]
    print('num peaks : ', len(peaks))

    try:
        target_depth = depth_hist[1][peaks[0]]
        print('target depth: ', target_depth)
    except:
        print('no peaks')

    depth_pixels = depth_crop.reshape(depth_crop.shape[0]*depth_crop.shape[1], 1)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(depth_pixels)
    # pic2show = kmeans.cluster_centers_[kmeans.labels_]
    pic2show = kmeans.labels_
    cluster_pic = pic2show.reshape(depth_crop.shape[0], depth_crop.shape[1])


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(depth)
    ax2.imshow(depth_crop)
    ax3.imshow(cluster_pic)
    ax4.plot(depth_hist[1][:-1], depth_hist[0])
    ax4.plot(depth_hist[1][peaks], depth_hist[0][peaks], 'x')
    plt.show()
