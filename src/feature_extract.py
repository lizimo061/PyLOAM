import numpy as np
import threading
import queue
import math

class FeatureExtract:
    def __init__(self, config=None):
        self.config = config
        self.LINE_NUM = 16
        self.RING_INDEX = 4 # None
        self.RING_INIT = True # False
        self.THRES = 0 # 2
        self.used_line_num = None
    
    def get_scan_id(self, cloud):
        xy_dist = np.sqrt(np.sum(np.square(cloud[:, :2]), axis=1))
        angles = np.arctan(cloud[:, 2]/xy_dist) * 180/math.pi
        if self.LINE_NUM == 16:
            scan_ids = (angles + 15)/2 + 0.5
        elif self.LINE_NUM == 32:
            scan_ids = int((angles + 93./3.) * 3./4.)
        elif self.LINE_NUM == 64:
            scan_ids = self.LINE_NUM / 2 + (-8.83 - angles) * 2. + .5
            upper = np.where(angles >= -8.83)
            scan_ids[upper] = (2 - angles[upper]) * 3.0 + 0.5
        else:
            print("Specific line number not supported!")
            return
        scan_ids = scan_ids.astype(int)
        if self.LINE_NUM == 64:
            correct_id = np.where(np.logical_and(scan_ids >= 0, scan_ids < 50)) # Only use 50 lines
        else:
            correct_id = np.where(np.logical_and(scan_ids >= 0, scan_ids < self.LINE_NUM))
        scan_ids = scan_ids[correct_id]
        cloud = cloud[correct_id]
        scan_ids = np.expand_dims(scan_ids, axis=1)
        return cloud, scan_ids    

    def remove_close_points(self, cloud, thres):
        """ Input size: N*3 """
        dists = np.sum(np.square(cloud[:, :3]), axis=1)
        cloud_out = cloud[dists > thres*thres]
        return cloud_out

    def divide_lines(self, cloud):
        line_num = np.max(cloud[:, self.RING_INDEX]) + 1
        self.used_line_num = int(line_num)
        clouds_by_line = [cloud[cloud[:, self.RING_INDEX] == val, :] for val in range(0, self.used_line_num)]
        cloud_out = np.concatenate(clouds_by_line, axis=0)
        return cloud_out

    def compute_curvatures(self, cloud):
        kernel = np.ones(11)
        kernel[5] = -10
        curvatures = np.apply_along_axis(lambda x: np.convolve(x, kernel, 'same'), 0, cloud[:, :3])
        curvatures = np.sum(np.square(curvatures), axis=1)
        scan_start_id = [np.where(cloud[:, self.RING_INDEX] == val)[0][0] + 5 for val in range(0, self.used_line_num)]
        scan_end_id = [np.where(cloud[:, self.RING_INDEX] == val)[0][-1] - 5 for val in range(0, self.used_line_num)]
        import pdb; pdb.set_trace()
        return curvatures, scan_start_id, scan_end_id

    def remove_occluded(self, cloud):
        num_points = cloud.shape[0]
        depth = np.sqrt(np.sum(np.square(cloud[:, :3]), axis=1))
        picked_list = np.zeros(num_points, dtype=int)
        for i in range(5, num_points-6):
            diff = np.sum(np.square(cloud[i, :3] - cloud[i+1, :3]))
            if diff > 0.1:
                if depth[i] > depth[i+1]:
                    depth_diff = cloud[i+1, :3] - cloud[i, :3] * (depth[i+1]/depth[i])
                    depth_diff = np.sqrt(np.sum(np.square(depth_diff)))
                    if depth_diff/depth[i+1] < 0.1:
                        picked_list[i-5:i+1] = 1
                else:
                    depth_diff = cloud[i+1, :3] * (depth[i]/depth[i+1]) - cloud[i, :3]
                    depth_diff = np.sqrt(np.sum(np.square(depth_diff)))
                    if depth_diff/depth[i] < 0.1:
                        picked_list[i+1:i+7] = 1

            diff_prev = np.sum(np.square(cloud[i, :3] - cloud[i-1, :3]))
            if diff > 0.0002 * depth[i] * depth[i] and diff_prev > 0.0002 * depth[i] * depth[i]:
                picked_list[i] = 1

        return picked_list

    def feature_classification(self, cloud, curvatures, picked_list, scan_start_id, scan_end_id):
        corner_sharp = []
        corner_less = []
        surf_flat = []
        surf_less = []
        cloud_labels = np.zeros(cloud.shape[0])
        index = np.arange(cloud.shape[0])
        index = np.expand_dims(index, axis=1).astype('float64')
        curvatures = np.expand_dims(curvatures, axis=1)

        curv_index = np.hstack((curvatures, index))
        for scan_id in range(self.used_line_num):
            """ TODO: Avoid empty line """
            for i in range(6):
                sp = int((scan_start_id[scan_id] * (6-i) + scan_end_id[scan_id] * i) / 6)
                ep = int((scan_start_id[scan_id] * (5-i) + scan_end_id[scan_id] * (i+1)) / 6 - 1)
                curv_seg = curv_index[sp:ep+1, :]
                sorted_curv = curv_seg[np.argsort(curv_seg[:, 0])]
                picked_num = 0

                for j in range(ep, sp-1, -1):
                    sorted_ind = j - sp
                    ind = int(sorted_curv[sorted_ind, 1])
                    curv = sorted_curv[sorted_ind, 0]
                    if picked_list[ind] == 0 and curv > 0.1:
                        picked_num += 1
                        if picked_num <= 2:
                            cloud_labels[ind] = 2
                            corner_sharp.append(ind)
                            corner_less.append(ind)
                        elif picked_num <= 20:
                            cloud_labels[ind] = 1
                            corner_less.append(ind)
                        else:
                            break

                        picked_list[ind] = 1

                        for l in range(1,6):
                            diff = np.sum(np.square(cloud[ind+l, :3] - cloud[ind+l-1, :3]))
                            if diff > 0.05:
                                break
                            picked_list[ind+l] = 1

                        for l in range(-1, -6, -1):
                            diff = np.sum(np.square(cloud[ind+l, :3] - cloud[ind+l+1, :3]))
                            if diff > 0.05:
                                break
                            picked_list[ind+l] = 1
                
                picked_num = 0
                for j in range(sp, ep+1):
                    sorted_ind = j - sp
                    ind = int(sorted_curv[sorted_ind, 1])
                    curv = sorted_curv[sorted_ind, 0]
                    if picked_list[ind] == 0 and curv < 0.1:
                        cloud_labels[ind] = -1
                        surf_flat.append(ind)
                        picked_num += 1

                        if picked_num >= 4:
                            break
                        
                        picked_list[ind] = 1

                        for l in range(1,6):
                            diff = np.sum(np.square(cloud[ind+l, :3] - cloud[ind+l-1, :3]))
                            if diff > 0.05:
                                break
                            picked_list[ind+l] = 1

                        for l in range(-1, -6, -1):
                            diff = np.sum(np.square(cloud[ind+l, :3] - cloud[ind+l+1, :3]))
                            if diff > 0.05:
                                break
                            picked_list[ind+l] = 1
                
                for j in range(sp, ep+1):
                    sorted_ind = j - sp
                    ind = int(sorted_curv[sorted_ind, 1])
                    if cloud_labels[ind] <= 0:
                        surf_less.append(ind)
        
        return corner_sharp, corner_less, surf_flat, surf_less

    def feature_extract(self, cloud):
        if self.RING_INIT is False:
            cloud, line_id = self.get_scan_id(cloud)
            cloud = np.hstack((cloud, line_id.astype(np.float32)))
            self.RING_INDEX = cloud.shape[1]-1

        cloud = self.remove_close_points(cloud, self.THRES)
        cloud = self.divide_lines(cloud)
        curvatures, scan_start_id, scan_end_id = self.compute_curvatures(cloud)
        picked_list = self.remove_occluded(cloud)
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_classification(cloud, curvatures, picked_list, scan_start_id, scan_end_id)
        return cloud[corner_sharp, :], cloud[corner_less, :], cloud[surf_flat, :], cloud[surf_less, :]
    
    def feature_extract_id(self, cloud):
        if self.RING_INIT is False:
            cloud, line_id = self.get_scan_id(cloud)
            cloud = np.hstack((cloud, line_id.astype(np.float32)))
            self.RING_INDEX = cloud.shape[1]-1

        cloud = self.remove_close_points(cloud, self.THRES)
        cloud = self.divide_lines(cloud)
        curvatures, scan_start_id, scan_end_id = self.compute_curvatures(cloud)
        picked_list = self.remove_occluded(cloud)
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_classification(cloud, curvatures, picked_list, scan_start_id, scan_end_id)
        return [corner_sharp, corner_less, surf_flat, surf_less]

class FeatureManager(threading.Thread):
    feature_queue = []
    process_id = 0
    process_lock = threading.Lock()

    def __init__(self, loader, config=None):
        threading.Thread.__init__(self)
        self.feature_extractor = FeatureExtract(config=config)
        self.data_loader = loader

    def run(self):
        while FeatureManager.process_id < len(self.data_loader):
            with FeatureManager.process_lock:
                feature_idx = self.feature_extractor.feature_extract_id(self.data_loader[FeatureManager.process_id])
                print("Feature processed: ", FeatureManager.process_id)
                FeatureManager.process_id += 1
                FeatureManager.feature_queue.append(feature_idx)

    @classmethod
    def get_feature(self):
        feature = []
        with FeatureManager.process_lock:
            if len(FeatureManager.feature_queue) is 0:
                print("Error: No feature processed")
            else:
                feature = FeatureManager.feature_queue.pop(0)
        return feature
