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
        self.rot_w_curr = np.eye(3)
        self.trans_w_curr = np.zeros((3,1))
        self.transform = np.array([0., 0., 0., 0., 0., 0.]) # rx, ry, rz, tx, ty, tz
        # TODO: make below variables to config
        self.DIST_THRES = 5
        self.RING_INDEX = 4
        self.NEARBY_SCAN = 2.5
        self.OPTIM_ITERATION = 25
        self.DISTORTION = False
        self.USE_ROBUST_LOSS = False
        self.VOXEL_SIZE = 0.2

        # For test,
        self.trans_list = []
    
    def angle_norm(self, angle):
        if angle <= -math.pi:
            angle += 2*math.pi
        elif angle > math.pi:
            angle -= 2*math.pi
        return angle

    def grab_frame(self, cloud):
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_extractor.feature_extract(cloud)
        # corner_sharp = cloud[feature_id[0], :]
        # corner_less = cloud[feature_id[1], :]
        # surf_flat = cloud[feature_id[2], :]
        # surf_less = cloud[feature_id[3], :]
        is_degenerate = False
        if not self.init:
            self.init = True
        else:
            weight = 1.0
            P_mat = np.identity(6)
            if self.USE_ROBUST_LOSS:
                loss = HuberLoss.Huber(1.0)
            else:
                loss = None
            for opt_iter in range(self.OPTIM_ITERATION):
                if opt_iter % 5 == 0:
                    corner_points, corner_points_a, corner_points_b = self.get_corner_correspondences(corner_sharp)
                    surf_points, surf_points_a, surf_points_b, surf_points_c = self.get_surf_correspondences(surf_flat)
                edge_A, edge_B = self.get_edge_mat(corner_points, corner_points_a, corner_points_b, 1.0)
                surf_A, surf_B = self.get_plane_mat(surf_points, surf_points_a, surf_points_b, surf_points_c, 1.0)

                A_mat = np.vstack((edge_A, surf_A))
                B_mat = np.vstack((edge_B, surf_B)) * -0.05 # Reference to original LOAM

                AtA = np.matmul(A_mat.transpose(), A_mat)
                AtB = np.matmul(A_mat.transpose(), B_mat)
                X_mat = np.linalg.solve(AtA, AtB)
                
                if opt_iter == 0:
                    vals, vecs = np.linalg.eig(AtA)
                    eigen_vec = vecs.copy()
                    for i in range(6):
                        if vals[i] < 10:
                            print("Warning: Degenerate!")
                            is_degenerate = True
                            eigen_vec[:, i] = np.zeros(6)
                        else:
                            break
                    P_mat = np.matmul(np.linalg.inv(vecs), eigen_vec)
                
                if is_degenerate:
                    X_mat = np.matmul(P_mat, X_mat)
                
                self.transform += np.squeeze(X_mat)

                delta_r = np.linalg.norm(np.rad2deg(X_mat[:3]))
                delta_t = np.linalg.norm(X_mat[4:] * 100)
                if delta_r < 0.1 and delta_t < 0.1:
                    print("Delta too small.")
                    break

            R_last_curr = get_rotation(self.transform[0], self.transform[1], self.transform[2])
            t_last_curr = -np.matmul(R_last_curr, self.transform[3:].reshape(3,1))
            self.trans_w_curr = self.trans_w_curr + np.matmul(self.rot_w_curr, t_last_curr)
            self.rot_w_curr = self.rot_w_curr * R_last_curr
            self.trans_list.append(self.trans_w_curr)

        surf_less_index = self.get_downsample_cloud(surf_less)
        corner_less_index = self.get_downsample_cloud(corner_less)
        self.surf_last = surf_less[surf_less_index, :]
        self.corner_last = corner_less[corner_less_index, :]
        self.transform = np.array([0., 0., 0., 0., 0., 0.])
        return self.trans_w_curr

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
                min_sq_dist2 = self.DIST_THRES * self.DIST_THRES

                for j in range(closest_ind+1, corner_sharp.shape[0]):
                    if self.corner_last[j, self.RING_INDEX] <= closest_scan_id:
                        continue
                    if self.corner_last[j, self.RING_INDEX] > closest_scan_id + self.NEARBY_SCAN:
                        break

                    point_sq_dist = np.sum(np.square(self.corner_last[j, :3].reshape(3,1) - point_sel))
                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j

                for j in range(closest_ind-1, -1, -1):
                    if self.corner_last[j, self.RING_INDEX] >= closest_scan_id:
                        continue
                    if self.corner_last[j, self.RING_INDEX] < closest_scan_id - self.NEARBY_SCAN:
                        break

                    point_sq_dist = np.sum(np.square(self.corner_last[j, :3].reshape(3,1) - point_sel))
                    if point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                
                if min_ind2 >= 0:
                    ab_dist = np.sum(np.square(self.surf_last[min_ind2, :3]-self.surf_last[closest_ind, :3]))
                    if ab_dist < 1e-3:
                        continue
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
                min_sq_dist2 = self.DIST_THRES * self.DIST_THRES
                min_sq_dist3 = self.DIST_THRES * self.DIST_THRES
                
                for j in range(closest_ind+1, surf_flat.shape[0]):
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
                    if self.surf_last[j, self.RING_INDEX] >= closest_scan_id and point_sq_dist < min_sq_dist2:
                        min_sq_dist2 = point_sq_dist
                        min_ind2 = j
                    elif self.surf_last[j, self.RING_INDEX] < closest_scan_id and point_sq_dist < min_sq_dist3:
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

    def get_downsample_cloud(self, cloud):
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3])
        max_bound = o3d_cloud.get_max_bound() + self.VOXEL_SIZE * 0.5
        min_bound = o3d_cloud.get_min_bound() - self.VOXEL_SIZE * 0.5
        out = o3d_cloud.voxel_down_sample_and_trace(self.VOXEL_SIZE, min_bound, max_bound, False)
        index_ds = [cubic_index[0] for cubic_index in out[2]]
        return index_ds

    def transform_to_start(self, pt):
        s = 1.0
        if self.DISTORTION:
            s = 0.5  # TODO: hard code
        scaled_transform = s * self.transform
        rot_mat = get_rotation(scaled_transform[0], scaled_transform[1], scaled_transform[2])
        translation = scaled_transform[3:6]
        undistorted_pt = np.transpose(rot_mat).dot(pt.reshape(3,1) - translation.reshape(3,1))
        return undistorted_pt

    def transfrom_to_end(self, pt):
        un_point = self.transform_to_start(pt)
        rot_mat = get_rotation(self.transform[0], self.transform[1], self.transform[2])
        translation = self.transform[3:6]
        pt_end = rot_mat.T.dot(un_point) + translation
        return pt_end

    def get_plane_mat(self, surf_points, surf_points_a, surf_points_b, surf_points_c, weight):
        A_mat = np.empty([len(surf_points), 6])
        B_mat = np.empty([len(surf_points), 1])

        srx = np.sin(self.transform[0])
        crx = np.cos(self.transform[0])
        sry = np.sin(self.transform[1])
        cry = np.cos(self.transform[1])
        srz = np.sin(self.transform[2])
        crz = np.cos(self.transform[2])
        tx = self.transform[3]
        ty = self.transform[4]
        tz = self.transform[5]
        for i in range(len(surf_points)):
            pt = surf_points[i].reshape(3,1)
            pt_a = surf_points_a[i].reshape(3,1)
            pt_b = surf_points_b[i].reshape(3,1)
            pt_c = surf_points_c[i].reshape(3,1)
            pt_sel = self.transform_to_start(pt)
            plane_norm = np.cross((pt_a - pt_b), (pt_a - pt_c), axis=0)
            norm = np.linalg.norm(plane_norm)
            plane_norm = plane_norm / norm

            if norm < 1e-5:
                continue

            B_mat[i, 0] = np.dot(np.transpose(plane_norm),(pt_sel - pt_a)) * weight
            A_mat[i, 0] = (-crx*sry*srz*pt[0] + crx*crz*sry*pt[1] + srx*sry*pt[2] \
                          + tx*crx*sry*srz - ty*crx*crz*sry - tz*srx*sry) * plane_norm[0] \
                          + (srx*srz*pt[0] - crz*srx*pt[1] + crx*pt[2] \
                          + ty*crz*srx - tz*crx - tx*srx*srz) * plane_norm[1] \
                          + (crx*cry*srz*pt[0] - crx*cry*crz*pt[1] - cry*srx*pt[2] \
                          + tz*cry*srx + ty*crx*cry*crz - tx*crx*cry*srz) * plane_norm[2]
            A_mat[i, 1] = ((-crz*sry - cry*srx*srz)*pt[0] \
                          + (cry*crz*srx - sry*srz)*pt[1] - crx*cry*pt[2] \
                          + tx*(crz*sry + cry*srx*srz) + ty*(sry*srz - cry*crz*srx) \
                          + tz*crx*cry) * plane_norm[0] \
                          + ((cry*crz - srx*sry*srz)*pt[0] \
                          + (cry*srz + crz*srx*sry)*pt[1] - crx*sry*pt[2] \
                          + tz*crx*sry - ty*(cry*srz + crz*srx*sry) \
                          - tx*(cry*crz - srx*sry*srz)) * plane_norm[2]
            A_mat[i, 2] = ((-cry*srz - crz*srx*sry)*pt[0] + (cry*crz - srx*sry*srz)*pt[1] \
                          + tx*(cry*srz + crz*srx*sry) - ty*(cry*crz - srx*sry*srz)) * plane_norm[0] \
                          + (-crx*crz*pt[0] - crx*srz*pt[1] \
                          + ty*crx*srz + tx*crx*crz) * plane_norm[1] \
                          + ((cry*crz*srx - sry*srz)*pt[0] + (crz*sry + cry*srx*srz)*pt[1] \
                          + tx*(sry*srz - cry*crz*srx) - ty*(crz*sry + cry*srx*srz)) * plane_norm[2]
            A_mat[i, 3] = -(cry*crz - srx*sry*srz) * plane_norm[0] + crx*srz * plane_norm[1] \
                          - (crz*sry + cry*srx*srz) * plane_norm[2]
            A_mat[i, 4] = -(cry*srz + crz*srx*sry) * plane_norm[0] - crx*crz * plane_norm[1] \
                          - (sry*srz - cry*crz*srx) * plane_norm[2]
            A_mat[i, 5] = crx*sry * plane_norm[0] - srx * plane_norm[1] - crx*cry * plane_norm[2]

        return A_mat, B_mat

    def get_edge_mat(self, corner_points, corner_points_a, corner_points_b, weight):
        A_mat = np.empty([len(corner_points), 6])
        B_mat = np.empty([len(corner_points), 1])

        srx = np.sin(self.transform[0])
        crx = np.cos(self.transform[0])
        sry = np.sin(self.transform[1])
        cry = np.cos(self.transform[1])
        srz = np.sin(self.transform[2])
        crz = np.cos(self.transform[2])
        tx = self.transform[3]
        ty = self.transform[4]
        tz = self.transform[5]

        for i in range(len(corner_points)):
            pt = corner_points[i].reshape(3,1)
            pt_a = corner_points_a[i].reshape(3,1)
            pt_b = corner_points_b[i].reshape(3,1)
            pt_sel = self.transform_to_start(pt)
            edge_normal = np.cross((pt_sel - pt_a), (pt_sel - pt_b), axis=0)
            ab = pt_a - pt_b
            ab_norm = np.linalg.norm(ab)
            edge_norm = np.linalg.norm(edge_normal)

            if edge_norm < 1e-5:
                continue

            la = (ab[1]*edge_normal[2] + ab[2]*edge_normal[1]) / (ab_norm*edge_norm)
            lb = -(ab[0]*edge_normal[2] - ab[2]*edge_normal[0]) / (ab_norm*edge_norm)
            lc = -(ab[0]*edge_normal[1] + ab[1]*edge_normal[0]) / (ab_norm*edge_norm)
            
            B_mat[i, 0] =  weight * (edge_norm / ab_norm)
            A_mat[i, 0] = (-crx*sry*srz*pt[0] + crx*crz*sry*pt[1] + srx*sry*pt[2] \
                          + tx*crx*sry*srz - ty*crx*crz*sry - tz*srx*sry) * la \
                          + (srx*srz*pt[0] - crz*srx*pt[1] + crx*pt[2] \
                          + ty*crz*srx - tz*crx - tx*srx*srz) * lb \
                          + (crx*cry*srz*pt[0] - crx*cry*crz*pt[1] - cry*srx*pt[2] \
                          + tz*cry*srx + ty*crx*cry*crz - tx*crx*cry*srz) * lc
            A_mat[i, 1] = ((-crz*sry - cry*srx*srz)*pt[0] \
                          + (cry*crz*srx - sry*srz)*pt[1] - crx*cry*pt[2] \
                          + tx*(crz*sry + cry*srx*srz) + ty*(sry*srz - cry*crz*srx) \
                          + tz*crx*cry) * la \
                          + ((cry*crz - srx*sry*srz)*pt[0] \
                          + (cry*srz + crz*srx*sry)*pt[1] - crx*sry*pt[2] \
                          + tz*crx*sry - ty*(cry*srz + crz*srx*sry) \
                          - tx*(cry*crz - srx*sry*srz)) * lc
            A_mat[i, 2] = ((-cry*srz - crz*srx*sry)*pt[0] + (cry*crz - srx*sry*srz)*pt[1] \
                          + tx*(cry*srz + crz*srx*sry) - ty*(cry*crz - srx*sry*srz)) * la \
                          + (-crx*crz*pt[0] - crx*srz*pt[1] \
                          + ty*crx*srz + tx*crx*crz) * lb \
                          + ((cry*crz*srx - sry*srz)*pt[0] + (crz*sry + cry*srx*srz)*pt[1] \
                          + tx*(sry*srz - cry*crz*srx) - ty*(crz*sry + cry*srx*srz)) * lc
            A_mat[i, 3] = -(cry*crz - srx*sry*srz) * la + crx*srz * lb \
                          - (crz*sry + cry*srx*srz) * lc
            A_mat[i, 4] = -(cry*srz + crz*srx*sry) * la - crx*crz * lb \
                          - (sry*srz - cry*crz*srx) * lc
            A_mat[i, 5] = crx*sry * la - srx * lb - crx*cry * lc

        return A_mat, B_mat

    def get_trans_list(self):
        return self.trans_list