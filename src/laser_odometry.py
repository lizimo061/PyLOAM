import numpy as np
import open3d as o3d
from feature_extract import FeatureExtract
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Odometry:
    def __init__(self, config=None):
        self.config = config
        self.init = False
        self.surf_last = None
        self.corner_last = None
        self.feature_extractor = FeatureExtract()
        self.q_w_curr = np.array([0, 0, 0, 1])
        self.t_w_curr = np.array([0, 0, 0])
        self.q_last_curr = np.array([0, 0, 0, 1])
        self.t_last_curr = np.array([0, 0, 0])
        # TODO: make below variables to config
        self.DIST_THRES = 25
        self.RING_INDEX = 4
        self.NEARBY_SCAN = 2.5
        self.OPTIM_ITERATION = 2
        self.DISTORTION = False

    def grab_frame(self, cloud):
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_extractor.feature_extract(cloud)
        if not self.init:
            self.init = True
            self.surf_last = surf_less
            self.corner_last = corner_less
        else:
            for opt_iter in range(self.OPTIM_ITERATION)

    def get_corner_correspondences(self, corner_sharp):
        curr_points = []
        points_a = []
        points_b = []
        corner_last_tree = o3d.geometry.KDTreeFlann(np_o3d(self.corner_last))

        for i in range(corner_sharp.shape[0]):
            point_sel = self.transform_to_start(corner_sharp[i, :3])
            _, ind, dist = corner_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1
            if dist[0] < self.DIST_THRES:
                closest_ind = ind
                closest_scan_id = self.corner_last[ind, self.RING_INDEX]
                min_sq_dist2 = self.DIST_THRES
                for j in range(closest_ind+1, self.corner_last.shape[0]):
                    if self.corner_last[j, self.RING_INDEX] <= closest_scan_id:
                        continue
                    if self.corner_last[j, self.RING_INDEX] > closest_scan_id + self.NEARBY_SCAN:
                        break

                    point_sq_dist = np.sum(np.square(self.corner_last[j, self.RING_INDEX] - point_sel))
                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j

                for j in range(closest_ind-1, -1, -1):
                    if self.corner_last[j, self.RING_INDEX] >= closest_scan_id:
                        continue
                    if self.corner_last[j, self.RING_INDEX] < closest_scan_id - self.NEARBY_SCAN:
                        break

                    point_sq_dist = np.sum(np.square(self.corner_last[j, self.RING_INDEX] - point_sel))
                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                
                if min_ind2 >= 0:
                    curr_points.append(corner_sharp[i, :3])
                    points_a.append(self.corner_last[closest_ind, :3])
                    points_b.append(self.corner_last[min_ind2, :3])
        
        return curr_points, points_a, points_b

    def get_surf_correspondences(self, surf_flat):
        curr_points = []
        points_a = []
        points_b = []
        points_c = []
        surf_last_tree = o3d.geometry.KDTreeFlann(np_o3d(self.surf_last))
        if i in range(surf_flat.shape[0]):
            point_sel = self.transform_to_start(surf_flat)
            _, ind, dist = surf_last_tree.KDTreeFlann(np_o3d(self.surf_last))
            closest_ind = -1
            min_ind2 = -1 
            min_ind3 = -1
            if dist[0] < self.DIST_THRES:
                closest_ind = ind
                closest_scan_id = self.corner_last[ind, self.RING_INDEX]
                min_sq_dist2 = self.DIST_THRES
                min_sq_dist3 = self.DIST_THRES
                for j in range(closest_ind+1, self.surf_last.shape[0]):
                    if self.surf_last[j, self.RING_INDEX] > closest_scan_id + self.NEARBY_SCAN:
                        break
                    point_sq_dist = np.sum(np.square(self.surf_last[j, self.RING_INDEX] - point_sel))
                    if self.surf_last[j, self.RING_INDEX] <= closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif self.surf_last[j, self.RING_INDEX] > closest_scan_id and point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j
                
                for j in range(closest_ind-1, -1, -1):
                    if self.surf_last[j, self.RING_INDEX] < closest_scan_id - self.NEARBY_SCAN:
                        break
                    point_sq_dist = np.sum(np.square(self.surf_last[j, self.RING_INDEX] - point_sel))
                    if self.surf_last[j, self.RING_INDEX] <= closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif self.surf_last[j, self.RING_INDEX] > closest_scan_id and point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j
                
                if min_ind2 >= 0 and min_ind3 >= 0:
                    curr_points.append(surf_flat[i, :3])
                    points_a.append(self.surf_last[closest_ind, :3])
                    points_b.append(self.surf_last[min_ind2, :3])
                    points_c.append(self.surf_last[min_ind3, :3])
                    
        return curr_points, points_a, points_b, points_c

    def transform_to_start(self, pt):
        s = 1.0
        if self.DISTORTION:
            s = 0.5  # TODO: hard code
        q_unit = np.array([0, 0, 0, 1])
        rots = R.from_quat([q_unit, self.q_last_curr])
        slerp = Slerp([0, 1], rots)
        rot_point_last = slerp([s])
        undistorted_pt = rot_point_last.as_dcm() * pt.reshape(3,1) + self.t_last_curr.reshape(3,1) * s
        return undistorted_pt

    @staticmethod
    def np_o3d(cloud):
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(cloud[:, :3])
        return pc_o3d


class EdgeErrorFactor(Factor):
    pass

class PlaneErrorFactor(Factor):
    pass