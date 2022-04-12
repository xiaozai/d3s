import os
import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt, peak_prominences, peak_widths
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
from scipy.cluster.hierarchy import fclusterdata

import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

'''
Code from:
https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c
'''
def chooseBestKforKMeans(scaled_data, k_range):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns
    -------
    scaled_inertia: float
        scaled inertia value for current k
    '''

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeansParallel(scaled_data, k_range):
    '''
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''

    ans = Parallel(n_jobs=-1,verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in k_range)
    ans = list(zip(k_range,ans))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

if __name__ == '__main__':

    data_path = '/home/yan/Data2/VOT/vot-workspace-CDTB/sequences/'
    sequences = os.listdir(data_path)
    try:
        sequences.remove('list.txt')
    except:
        pass

    for seq in sequences[:20]:
    # seq = 'box_darkroom_noocc_2'
        print(seq)
        depth_path = data_path + seq + '/' + 'depth/'
        gt_path = data_path + seq + '/' 'groundtruth.txt'

        with open(gt_path, 'r') as fp:
            boxes = fp.readlines()
        boxes = [box.strip() for box in boxes]

        tic = time.time()
        frame_id = 1
        depth = cv2.imread(depth_path+'%08d.png'%frame_id, -1)
        box = boxes[frame_id-1]
        box = [int(float(b)) for b in box.split(',')]
        num_pixels = box[2]*box[3]
        print('num pixels: ', num_pixels)

        depth_crop = np.asarray(depth[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
        print('max depth:', np.max(depth_crop), '  min depth: ', np.min(depth_crop))

        depth_crop = medfilt(depth_crop, 15)
        depth_hist, depth_edges = np.histogram(depth_crop.flatten(), bins=20)
        hist_bins = (depth_edges[:-1] + depth_edges[1:]) / 2.0

        peaks, properties = find_peaks(depth_hist, height=num_pixels/10)
        print('num peaks : ', len(peaks))

        prominences, left_bases, right_bases = peak_prominences(depth_hist, peaks)
        print(prominences)
        print(left_bases, right_bases)

        try:
            target_depth = hist_bins[peaks[0]]
            print('target depth: ', target_depth)

            if len(peaks) > 1:
                left = left_bases[0]
                right = min(right_bases[0], left_bases[1])
            else:
                left = left_bases[0]
                right = right_bases[0]
            print(left, right)

            max_depth, min_depth = hist_bins[right], hist_bins[left]
            print('depth range: ', min_depth, max_depth)

        except:
            print('no peaks')
            target_depth = np.median(depth_crop[depth_crop>0])
            print('target depth:', target_depth)
            max_depth, min_depth = np.max(depth_crop.flatten()), np.min(depth_crop.flatten())
            left = 0
            right = 0


        depth_pixels = depth_crop.reshape(depth_crop.shape[0]*depth_crop.shape[1], 1)

        # fkmeans = fclusterdata(depth_pixels, t=50, criterion='distance')
        # fkmeans = fkmeans.reshape(depth_crop.shape[0], depth_crop.shape[1])
        # print(fkmeans)

        # compute adjusted intertia
        k_range = range(2,6)
        mms = MinMaxScaler()
        scaled_data = mms.fit_transform(depth_pixels)
        best_k, results = chooseBestKforKMeansParallel(scaled_data, k_range)
        print('best k:', best_k)
        print('results: ', results)

        kmeans = KMeans(n_clusters=best_k, random_state=0).fit(depth_pixels)
        print(kmeans.cluster_centers_)
        cluster_centers = kmeans.cluster_centers_
        cluster_dist = [abs(c - target_depth) for c in cluster_centers]
        target_label = cluster_dist.index(min(cluster_dist))
        kmeans_segm = kmeans.labels_ == target_label
        kmeans_segm = kmeans_segm.reshape(depth_crop.shape[0], depth_crop.shape[1])

        pic2show = kmeans.cluster_centers_[kmeans.labels_]
        cluster_pic = pic2show.reshape(depth_crop.shape[0], depth_crop.shape[1])
        # pic2show = kmeans.labels_
        # cluster_pic = pic2show.reshape(depth_crop.shape[0], depth_crop.shape[1])

        depth_segm = np.array(depth_crop, copy=True)
        depth_segm[depth_segm > max_depth] = 0
        depth_segm[depth_segm < min_depth] = 0
        depth_segm[depth_segm > 0] = 1
        depth_segm = np.array(depth_segm, dtype=np.uint8)

        toc = time.time()
        print('time:', toc-tic)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
        ax1.imshow(depth)
        ax2.imshow(depth_crop)
        ax3.imshow(cluster_pic)
        ax4.imshow(kmeans_segm)
        ax4.set_title('kmeans segm')
        ax5.plot(hist_bins, depth_hist)
        ax5.plot(hist_bins[peaks], depth_hist[peaks], 'x')
        ax5.axvline(x=hist_bins[left], color='r')
        ax5.axvline(x=hist_bins[right], color='b')

        ax6.imshow(depth_segm)

        plt.show()
