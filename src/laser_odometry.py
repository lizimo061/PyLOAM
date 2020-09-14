import numpy as np
import open3d as o3d
from feature_extract import FeatureExtract
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from minisam import *
from utils import *
import math

class Odometry:
    def __init__(self, config=None):
        self.config = config
        self.init = False
        self.surf_last = None
        self.corner_last = None
        self.feature_extractor = FeatureExtract()

        self.transform = np.array([0, 0, 0, 0, 0, 0]) # rx, ry, rz, tx, ty, tz
        # TODO: make below variables to config
        self.DIST_THRES = 25
        self.RING_INDEX = 4
        self.NEARBY_SCAN = 2.5
        self.OPTIM_ITERATION = 2
        self.DISTORTION = False
        self.USE_ROBUST_LOSS = True
    
    def angle_norm(self, angle):
        if angle <= -math.pi:
            angle += math.pi
        elif angle > math.pi:
            angle -= math.pi
        return angle

    def grab_frame(self, cloud):
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_extractor.feature_extract(cloud)
        if not self.init:
            self.init = True
            self.surf_last = surf_less
            self.corner_last = corner_less
        else:
            weight = 1.0
            if self.USE_ROBUST_LOSS:
                loss = HuberLoss.Huber(1.0)
            else:
                loss = None
            for opt_iter in range(self.OPTIM_ITERATION):
                single_pose_graph = FactorGraph()
                # corner_points, corner_points_a, corner_points_b = self.get_corner_correspondences(corner_sharp)
                surf_points, surf_points_a, surf_points_b, surf_points_c = self.get_surf_correspondences(surf_flat)
                print("Surf points: ", len(surf_points))
                for i in range(len(surf_points)):
                    single_pose_graph.add(PlaneFactor(key('p', 0), surf_points[i], surf_points_a[i], surf_points_b[i], surf_points_c[i], weight, loss))
                init_pose = Variables()
                init_pose.add(key('p', 0), self.transform)

                opt_param = LevenbergMarquardtOptimizerParams()
                opt_param.max_iterations = 3
                opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
                opt = LevenbergMarquardtOptimizer(opt_param)
                opt_pose = Variables()
                status = opt.optimize(single_pose_graph, init_pose, opt_pose)

                if status is NonlinearOptimizationStatus.SUCCESS:
                    print("Optimiazation error: ", status)
                print("Optimizied values: ", opt_pose.at(key('p', 0)))
                self.transform = opt_pose.at(key('p', 0)).copy()
                self.transform[0] = self.angle_norm(self.transform[0])
                self.transform[1] = self.angle_norm(self.transform[1])
                self.transform[2] = self.angle_norm(self.transform[2])

    def get_corner_correspondences(self, corner_sharp):
        curr_points = []
        points_a = []
        points_b = []
        corner_last_tree = o3d.geometry.KDTreeFlann(np.transpose(self.corner_last[:, :3]))

        for i in range(corner_sharp.shape[0]):
            point_sel = self.transform_to_start(corner_sharp[i, :3])
            [_, ind, dist] = corner_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1
            if dist[0] < self.DIST_THRES:
                closest_ind = ind[0]
                closest_scan_id = self.corner_last[ind[0], self.RING_INDEX]
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
        surf_last_tree = o3d.geometry.KDTreeFlann(np.transpose(self.surf_last[:, :3]))
        for i in range(surf_flat.shape[0]):
            point_sel = self.transform_to_start(surf_flat[i,:3])
            [_, ind, dist] = surf_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1 
            min_ind3 = -1
            if dist[0] < self.DIST_THRES:
                closest_ind = ind[0]
                closest_scan_id = self.surf_last[ind[0], self.RING_INDEX]
                min_sq_dist2 = self.DIST_THRES
                min_sq_dist3 = self.DIST_THRES
                
                for j in range(closest_ind+1, self.surf_last.shape[0]):
                    if self.surf_last[j, self.RING_INDEX] > closest_scan_id + self.NEARBY_SCAN:
                        break
                    point_sq_dist = np.sum(np.square(self.surf_last[j, :3].reshape(3,1) - point_sel))
                    if self.surf_last[j, self.RING_INDEX] <= closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif self.surf_last[j, self.RING_INDEX] > closest_scan_id and point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j
                
                for j in range(closest_ind-1, -1, -1):
                    if self.surf_last[j, self.RING_INDEX] < closest_scan_id - self.NEARBY_SCAN:
                        break
                    point_sq_dist = np.sum(np.square(self.surf_last[j, :3].reshape(3,1) - point_sel))
                    if self.surf_last[j, self.RING_INDEX] <= closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif self.surf_last[j, self.RING_INDEX] > closest_scan_id and point_sq_dist < min_sq_dist3:
                        min_sq_dist3 = point_sq_dist
                        min_ind3 = j
                
                if min_ind2 >= 0 and min_ind3 >= 0:
                    ab_dist = np.sum(np.square(self.surf_last[min_ind2, :3]-self.surf_last[closest_ind, :3]))
                    ac_dist = np.sum(np.square(self.surf_last[min_ind3, :3]-self.surf_last[closest_ind, :3]))
                    if ab_dist < 1e-3 or ac_dist < 1e-3:
                        continue
                    curr_points.append(surf_flat[i, :3])
                    points_a.append(self.surf_last[closest_ind, :3])
                    points_b.append(self.surf_last[min_ind2, :3])
                    points_c.append(self.surf_last[min_ind3, :3])

        return curr_points, points_a, points_b, points_c

    def transform_to_start(self, pt):
        s = 1.0
        if self.DISTORTION:
            s = 0.5  # TODO: hard code
        scaled_transform = s * self.transform
        rot_mat = get_rotation(scaled_transform[0], scaled_transform[1], scaled_transform[2])
        translation = scaled_transform[3:6]
        undistorted_pt = np.transpose(rot_mat).dot(pt.reshape(3,1) - translation.reshape(3,1))
        return undistorted_pt

class PlaneFactor(Factor):
    def __init__(self, key, surf_pt, pt_a, pt_b, pt_c, weight, loss):
        Factor.__init__(self, 1, [key], loss)
        self.surf_p_ = surf_pt.reshape(3,1)
        self.p_a_ = pt_a.reshape(3,1)
        self.p_b_ = pt_b.reshape(3,1)
        self.p_c_ = pt_c.reshape(3,1)
        self.weight = weight
        self.plane_norm = np.cross((self.p_a_ - self.p_b_), (self.p_a_ - self.p_c_), axis=0)
        norm = np.linalg.norm(self.plane_norm)
        self.plane_norm = self.plane_norm / norm
        

    def transform_curr(self, transform):
        scaled_transform = self.weight * transform
        rot_mat = get_rotation(scaled_transform[0], scaled_transform[1], scaled_transform[2])
        translation = scaled_transform[3:6]
        undistorted_pt = np.transpose(rot_mat).dot(self.surf_p_ - translation.reshape(3,1))
        return undistorted_pt
    
    def copy(self):
        return PlaneFactor(self.keys()[0], self.surf_p_, self.p_a_, self.p_b_, self.p_c_, self.weight, self.lossFunction())
    
    def error(self, variables):
        params = variables.at(self.keys()[0])
        point_sel = self.transform_curr(params)
        dist = np.dot(np.transpose(self.plane_norm),(point_sel - self.p_a_)) * self.weight
        return np.array([dist[0][0]])

    def jacobians(self, variables):
        params = variables.at(self.keys()[0])
        
        srx = np.sin(params[0])
        crx = np.cos(params[0])
        sry = np.sin(params[1])
        cry = np.cos(params[1])
        srz = np.sin(params[2])
        crz = np.cos(params[2])
        tx = params[3]
        ty = params[4]
        tz = params[5]

        J_d_transform = np.empty([1,6])
        J_d_transform[0][0] = (-crx*sry*srz*self.surf_p_[0] + crx*crz*sry*self.surf_p_[1] + srx*sry*self.surf_p_[2] \
                              + tx*crx*sry*srz - ty*crx*crz*sry - tz*srx*sry) * self.plane_norm[0] \
                              + (srx*srz*self.surf_p_[0] - crz*srx*self.surf_p_[1] + crx*self.surf_p_[2] \
                              + ty*crz*srx - tz*crx - tx*srx*srz) * self.plane_norm[1] \
                              + (crx*cry*srz*self.surf_p_[0] - crx*cry*crz*self.surf_p_[1] - cry*srx*self.surf_p_[2] \
                              + tz*cry*srx + ty*crx*cry*crz - tx*crx*cry*srz) * self.plane_norm[2]

        J_d_transform[0][1] = ((-crz*sry - cry*srx*srz)*self.surf_p_[0] \
                              + (cry*crz*srx - sry*srz)*self.surf_p_[1] - crx*cry*self.surf_p_[2] \
                              + tx*(crz*sry + cry*srx*srz) + ty*(sry*srz - cry*crz*srx) \
                              + tz*crx*cry) * self.plane_norm[0] \
                              + ((cry*crz - srx*sry*srz)*self.surf_p_[0] \
                              + (cry*srz + crz*srx*sry)*self.surf_p_[1] - crx*sry*self.surf_p_[2] \
                              + tz*crx*sry - ty*(cry*srz + crz*srx*sry) \
                              - tx*(cry*crz - srx*sry*srz)) * self.plane_norm[2]

        J_d_transform[0][2] = ((-cry*srz - crz*srx*sry)*self.surf_p_[0] + (cry*crz - srx*sry*srz)*self.surf_p_[1] \
                              + tx*(cry*srz + crz*srx*sry) - ty*(cry*crz - srx*sry*srz)) * self.plane_norm[0] \
                              + (-crx*crz*self.surf_p_[0] - crx*srz*self.surf_p_[1] \
                              + ty*crx*srz + tx*crx*crz) * self.plane_norm[1] \
                              + ((cry*crz*srx - sry*srz)*self.surf_p_[0] + (crz*sry + cry*srx*srz)*self.surf_p_[1] \
                              + tx*(sry*srz - cry*crz*srx) - ty*(crz*sry + cry*srx*srz)) * self.plane_norm[2]
        
        J_d_transform[0][3] = -(cry*crz - srx*sry*srz) * self.plane_norm[0] + crx*srz * self.plane_norm[1] \
                              - (crz*sry + cry*srx*srz) * self.plane_norm[2]
        J_d_transform[0][4] = -(cry*srz + crz*srx*sry) * self.plane_norm[0] - crx*crz * self.plane_norm[1] \
                              - (sry*srz - cry*crz*srx) * self.plane_norm[2]
        J_d_transform[0][5] = crx*sry * self.plane_norm[0] - srx * self.plane_norm[1] - crx*cry * self.plane_norm[2]
        return [J_d_transform]



