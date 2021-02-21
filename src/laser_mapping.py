import numpy as np
import open3d as o3d
from utils import *

class Mapper:
    def __init__(self, config=None):
        self.rot_wmap_wodom = np.eye(3)
        self.trans_wmap_wodom = np.zeros((3,1))
        self.frame_count = 0
        if config is None:
            self.CLOUD_WIDTH = 21
            self.CLOUD_HEIGHT = 21
            self.CLOUD_DEPTH = 11
            self.CORNER_VOXEL_SIZE = 0.2
            self.SURF_VOXEL_SIZE = 0.4
            self.MAP_VOXEL_SIZE = 0.6
            self.STACK_NUM = 2
            self.OPTIM_ITERATION = 5
        else:
            self.CLOUD_WIDTH = config['mapping']['cloud_width']
            self.CLOUD_HEIGHT = config['mapping']['cloud_height']
            self.CLOUD_DEPTH = config['mapping']['cloud_depth']
            self.CORNER_VOXEL_SIZE = config['mapping']['corner_voxel_size']
            self.SURF_VOXEL_SIZE = config['mapping']['surf_voxel_size']
            self.MAP_VOXEL_SIZE = config['mapping']['map_voxel_size']
            self.STACK_NUM = config['mapping']['stack_num']
            self.OPTIM_ITERATION = config['mapping']['optim_iteration']

        self.CUBE_NUM = self.CLOUD_DEPTH * self.CLOUD_HEIGHT * self.CLOUD_WIDTH

        self.cloud_center_width = int(self.CLOUD_WIDTH/2)
        self.cloud_center_height = int(self.CLOUD_HEIGHT/2)
        self.cloud_center_depth = int(self.CLOUD_DEPTH/2)
        self.valid_index = [-1] * 125
        self.surround_index = [-1] * 125

        self.cloud_corner_array = [[] for i in range(self.CUBE_NUM)]
        self.cloud_surf_array = [[] for i in range(self.CUBE_NUM)]
        self.frame_count = 0

        self.rot_wodom_curr = np.eye(3)
        self.trans_wodom_curr = np.zeros((3,1))
        self.rot_wmap_wodom = np.eye(3)
        self.trans_wmap_wodom = np.zeros((3,1))
        self.rot_w_curr = np.eye(3)
        self.trans_w_curr = np.zeros((3,1))
        self.transform = np.zeros(6)

    def transform_associate_to_map(self, rot_wodom_curr, trans_wodom_curr):
        self.rot_wodom_curr = rot_wodom_curr
        self.trans_wodom_curr = trans_wodom_curr
        self.rot_w_curr = np.matmul(self.rot_wmap_wodom, rot_wodom_curr)
        self.trans_w_curr = np.matmul(self.rot_wmap_wodom, trans_wodom_curr).reshape(3,1) + self.trans_wmap_wodom
        rx, ry, rz = get_euler_angles(self.rot_w_curr)
        self.transform = np.array([rx, ry, rz, self.trans_w_curr[0][0], self.trans_w_curr[1][0], self.trans_w_curr[2][0]])

    def point_associate_to_map(self, pt):
        pt_out = np.matmul(self.rot_w_curr, pt.T).T + self.trans_w_curr.reshape(1, 3)
        return pt_out.squeeze()
    
    def transform_update(self):
        self.rot_wmap_wodom = np.matmul(self.rot_w_curr, self.rot_wodom_curr.T)
        self.trans_wmap_wodom = self.trans_w_curr - np.matmul(self.rot_wmap_wodom, self.trans_wodom_curr.reshape(3,1))
    
    def transform_convert(self):
        self.rot_w_curr = get_rotation(self.transform[0], self.transform[1], self.transform[2])
        self.trans_w_curr = self.transform[3:].reshape(3,1)

    def map_frame(self, odom, corner_last, surf_last):
        print('Mapping frame: ', self.frame_count)
        if self.frame_count % self.STACK_NUM != 0:
            self.frame_count += 1
            return None
        rot_wodom_curr = odom[:3, :3]
        trans_wodom_curr = odom[:3, 3].reshape(3,1)
        self.transform_associate_to_map(rot_wodom_curr, trans_wodom_curr)
        cube_center_i = int((self.trans_w_curr[0][0] + 25.0) / 50.0) + self.cloud_center_width
        cube_center_j = int((self.trans_w_curr[1][0] + 25.0) / 50.0) + self.cloud_center_height
        cube_center_k = int((self.trans_w_curr[2][0] + 25.0) / 50.0) + self.cloud_center_depth

        if self.trans_w_curr[0][0] + 25.0 < 0:
            cube_center_i -= 1
        if self.trans_w_curr[1][0] + 25.0 < 0:
            cube_center_j -= 1
        if self.trans_w_curr[2][0] + 25.0 < 0:
            cube_center_k -= 1

        is_degenerate = False
        
        while cube_center_i < 3:
            for j in range(self.CLOUD_HEIGHT):
                for k in range(self.CLOUD_DEPTH):
                    i = self.CLOUD_WIDTH - 1
                    ind = j * self.CLOUD_WIDTH + k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                    corner_tmp = self.cloud_corner_array[i+ind]
                    surf_tmp = self.cloud_surf_array[i+ind]

                    for tmp in range(i, 0, -1):
                        self.cloud_corner_array[tmp+ind] = self.cloud_corner_array[tmp+ind-1]
                        self.cloud_surf_array[tmp+ind] = self.cloud_surf_array[tmp+ind-1]
                    self.cloud_corner_array[ind] = corner_tmp
                    self.cloud_surf_array[ind] = surf_tmp

            cube_center_i += 1
            self.cloud_center_width += 1
        
        while cube_center_i >= self.CLOUD_WIDTH - 3:
            for j in range(self.CLOUD_HEIGHT):
                for k in range(self.CLOUD_DEPTH):
                    i = 0
                    ind = j * self.CLOUD_WIDTH + k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                    corner_tmp = self.cloud_corner_array[i+ind]
                    surf_tmp = self.cloud_surf_array[i+ind]

                    for tmp in range(0, self.CLOUD_WIDTH-1):
                        self.cloud_corner_array[tmp+ind] = self.cloud_corner_array[tmp+ind+1]
                        self.cloud_surf_array[tmp+ind] = self.cloud_surf_array[tmp+ind+1]
                    self.cloud_corner_array[ind + self.CLOUD_WIDTH - 1] = corner_tmp
                    self.cloud_surf_array[ind + self.CLOUD_WIDTH - 1] = surf_tmp

            cube_center_i -= 1
            self.cloud_center_width -= 1

        while cube_center_j < 3:
            for i in range(self.CLOUD_WIDTH):
                for k in range(self.CLOUD_DEPTH):
                    j = self.CLOUD_HEIGHT - 1
                    ind = i + k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                    corner_tmp = self.cloud_corner_array[j * self.CLOUD_WIDTH + ind]
                    surf_tmp = self.cloud_surf_array[j * self.CLOUD_WIDTH + ind]

                    for tmp in range(j, 0, -1):
                        self.cloud_corner_array[tmp*self.CLOUD_WIDTH+ind] = self.cloud_corner_array[(tmp-1)*self.CLOUD_WIDTH+ind]
                        self.cloud_surf_array[tmp*self.CLOUD_WIDTH+ind] = self.cloud_surf_array[(tmp-1)*self.CLOUD_WIDTH+ind]
                    self.cloud_corner_array[ind] = corner_tmp
                    self.cloud_surf_array[ind] = surf_tmp

            cube_center_j += 1
            self.cloud_center_height += 1
        
        while cube_center_j >= self.CLOUD_HEIGHT - 3:
            for i in range(self.CLOUD_WIDTH):
                for k in range(self.CLOUD_DEPTH):
                    j = 0
                    ind = i + k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                    corner_tmp = self.cloud_corner_array[j*self.CLOUD_WIDTH+ind]
                    surf_tmp = self.cloud_surf_array[j*self.CLOUD_WIDTH+ind]

                    for tmp in range(0, self.CLOUD_HEIGHT-1):
                        self.cloud_corner_array[tmp*self.CLOUD_WIDTH+ind] = self.cloud_corner_array[(tmp+1)*self.CLOUD_WIDTH+ind]
                        self.cloud_surf_array[tmp*self.CLOUD_WIDTH+ind] = self.cloud_corner_array[(tmp+1)*self.CLOUD_WIDTH+ind]
                    self.cloud_corner_array[ind + (self.CLOUD_HEIGHT-1)*self.CLOUD_WIDTH] = corner_tmp
                    self.cloud_surf_array[ind + (self.CLOUD_HEIGHT-1)*self.CLOUD_WIDTH] = surf_tmp
  
            cube_center_j -= 1
            self.cloud_center_height -= 1

        while cube_center_k < 3:
            for i in range(self.CLOUD_WIDTH):
                for j in range(self.CLOUD_HEIGHT):
                    k = self.CLOUD_DEPTH - 1
                    ind = i + j * self.CLOUD_WIDTH
                    corner_tmp = self.cloud_corner_array[k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]
                    surf_tmp = self.cloud_surf_array[k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]

                    for tmp in range(k, 0, -1):
                        self.cloud_corner_array[tmp * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind] = self.cloud_corner_array[(tmp-1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]
                        self.cloud_surf_array[tmp * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind] = self.cloud_surf_array[(tmp-1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]

                    self.cloud_corner_array[ind] = corner_tmp
                    self.cloud_surf_array[ind] = surf_tmp

            cube_center_k += 1
            self.cloud_center_depth += 1
        
        while cube_center_k >= self.CLOUD_DEPTH - 3:
            for i in range(self.CLOUD_WIDTH):
                for j in range(self.CLOUD_HEIGHT):
                    k = 0
                    ind = i + j * self.CLOUD_WIDTH
                    corner_tmp = self.cloud_corner_array[k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]
                    surf_tmp = self.cloud_surf_array[k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]

                    for tmp in range(0, self.CLOUD_DEPTH - 1):
                        self.cloud_corner_array[tmp * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind] = self.cloud_corner_array[(tmp+1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]
                        self.cloud_surf_array[tmp * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind] = self.cloud_surf_array[(tmp+1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT + ind]
                    self.cloud_corner_array[ind + (self.CLOUD_DEPTH-1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT] = corner_tmp
                    self.cloud_surf_array[ind + (self.CLOUD_DEPTH-1) * self.CLOUD_WIDTH * self.CLOUD_HEIGHT] = surf_tmp
            
            cube_center_k -= 1
            self.cloud_center_depth -= 1
        
        valid_cloud_num = 0
        surround_cloud_num = 0

        for i in range(cube_center_i - 2, cube_center_i + 3):
            for j in range(cube_center_j - 2, cube_center_j + 3):
                for k in range(cube_center_k - 1, cube_center_k + 2):
                    if i>=0 and i<self.CLOUD_WIDTH and j>=0 and j<self.CLOUD_HEIGHT and k>=0 and k<self.CLOUD_DEPTH:
                        self.valid_index[valid_cloud_num] = i + j * self.CLOUD_WIDTH + k * self.CLOUD_HEIGHT * self.CLOUD_WIDTH
                        valid_cloud_num += 1
                        self.surround_index[surround_cloud_num] = i + j * self.CLOUD_WIDTH + k * self.CLOUD_HEIGHT * self.CLOUD_WIDTH
                        surround_cloud_num += 1
        
        map_corner_list = []
        map_surf_list = []
        for i in range(valid_cloud_num):
            map_corner_list += self.cloud_corner_array[self.valid_index[i]]
            map_surf_list += self.cloud_surf_array[self.valid_index[i]]
        
        if len(map_corner_list) > 0:
            corner_from_map = np.vstack(map_corner_list)

        if len(map_surf_list) > 0:
            surf_from_map = np.vstack(map_surf_list)
        
        _, corner_last_ds = downsample_filter(corner_last, self.CORNER_VOXEL_SIZE)
        _, surf_last_ds = downsample_filter(surf_last, self.SURF_VOXEL_SIZE)

        if len(map_corner_list) > 0 and len(map_surf_list) > 0 and corner_from_map.shape[0] > 10 and surf_from_map.shape[0] > 100:
            corner_map_tree = o3d.geometry.KDTreeFlann(np.transpose(corner_from_map[:, :3]))
            surf_map_tree = o3d.geometry.KDTreeFlann(np.transpose(surf_from_map[:, :3]))

            for iter_num in range(self.OPTIM_ITERATION):
                coeff_list = []
                pt_list = []
                
                # Find corner correspondences
                for i in range(corner_last_ds.shape[0]):
                    point_sel = self.point_associate_to_map(corner_last_ds[i, :3])
                    [_, ind, dist] = corner_map_tree.search_knn_vector_3d(point_sel, 5)
                    if dist[4] < 1.0:
                        center, cov = get_mean_cov(corner_from_map[ind, :3])
                        vals, vecs = np.linalg.eig(cov)
                        idx = vals.argsort() # Sort ascending
                        vals = vals[idx]
                        vecs = vecs[:, idx]

                        if vals[2] > 3 * vals[1]:
                            point_a = center + 0.1 * vecs[:, 2]
                            point_b = center - 0.1 * vecs[:, 2]
                            edge_normal = np.cross((point_sel - point_a), (point_sel - point_b), axis=0)
                            edge_norm = np.linalg.norm(edge_normal)
                            ab = point_a - point_b
                            ab_norm = np.linalg.norm(ab)

                            la = (ab[1]*edge_normal[2] + ab[2]*edge_normal[1]) / (ab_norm*edge_norm)
                            lb = -(ab[0]*edge_normal[2] + ab[2]*edge_normal[0]) / (ab_norm*edge_norm)
                            lc = (ab[0]*edge_normal[1] - ab[1]*edge_normal[0]) / (ab_norm*edge_norm)

                            ld = edge_norm / ab_norm

                            s = 1 - 0.9 * np.abs(ld)

                            if s > 0.1:
                                pt_list.append(corner_last_ds[i, :3])
                                coeff_list.append(np.array([s*la, s*lb, s*lc, s*ld]))

                # Find surface correspondences
                for i in range(surf_last_ds.shape[0]):
                    point_sel = self.point_associate_to_map(surf_last_ds[i, :3])
                    [_, ind, dist] = surf_map_tree.search_knn_vector_3d(point_sel, 5)

                    if dist[4] < 1.0:
                        surf_normal, _, _, _ = np.linalg.lstsq(surf_from_map[ind, :3], -np.ones((5,1)), rcond=None)
                        surf_norm = np.linalg.norm(surf_normal)
                        coeff = np.append(surf_normal, 1.0) / surf_norm

                        surf_homo = np.concatenate((surf_from_map[ind, :3], np.ones((5,1))), axis=1)
                        plane_residual = np.abs(np.matmul(surf_homo, coeff.reshape(4,1)))

                        if np.any(plane_residual > 0.2):
                            continue
                        
                        pd2 = np.dot(np.append(point_sel, 1), coeff)
                        s = 1 - 0.9 * np.abs(pd2) / np.sqrt(np.linalg.norm(point_sel))

                        coeff[3] = pd2
                        coeff = s*coeff

                        if s > 0.1:
                            coeff_list.append(coeff)
                            pt_list.append(surf_last_ds[i, :3])

                if len(coeff_list) < 50:
                    print("Warning: Few matches")
                    continue

                srx = np.sin(self.transform[0])
                crx = np.cos(self.transform[0])
                sry = np.sin(self.transform[1])
                cry = np.cos(self.transform[1])
                srz = np.sin(self.transform[2])
                crz = np.cos(self.transform[2])

                A_mat = []
                B_mat = []

                for i in range(len(coeff_list)):
                    A_tmp = np.zeros((1,6))
                    B_tmp = np.zeros((1,1))
                    A_tmp[0, 0] = (crx*sry*srz*pt_list[i][0] + crx*crz*sry*pt_list[i][1] - srx*sry*pt_list[i][2]) * coeff_list[i][0] \
                        + (-srx*srz*pt_list[i][0] - crz*srx*pt_list[i][1] - crx*pt_list[i][2]) * coeff_list[i][1] \
                        + (crx*cry*srz*pt_list[i][0] + crx*cry*crz*pt_list[i][1] - cry*srx*pt_list[i][2]) * coeff_list[i][2]

                    A_tmp[0, 1] = ((cry*srx*srz - crz*sry)*pt_list[i][0] \
                        + (sry*srz + cry*crz*srx)*pt_list[i][1] + crx*cry*pt_list[i][2]) * coeff_list[i][0] \
                        + ((-cry*crz - srx*sry*srz)*pt_list[i][0] \
                        + (cry*srz - crz*srx*sry)*pt_list[i][1] - crx*sry*pt_list[i][2]) * coeff_list[i][2]

                    A_tmp[0, 2] = ((crz*srx*sry - cry*srz)*pt_list[i][0] + (-cry*crz-srx*sry*srz)*pt_list[i][1])*coeff_list[i][0] \
                        + (crx*crz*pt_list[i][0] - crx*srz*pt_list[i][1]) * coeff_list[i][1] \
                        + ((sry*srz + cry*crz*srx)*pt_list[i][0] + (crz*sry-cry*srx*srz)*pt_list[i][1])*coeff_list[i][2]
                    
                    A_tmp[0, 3] = coeff_list[i][0]
                    A_tmp[0, 4] = coeff_list[i][1]
                    A_tmp[0, 5] = coeff_list[i][2]

                    B_tmp[0,0] = -coeff_list[i][3]

                    A_mat.append(A_tmp)
                    B_mat.append(B_tmp)

                A_mat = np.vstack(A_mat)
                B_mat = np.vstack(B_mat)
                AtA = np.matmul(A_mat.T, A_mat)
                AtB = np.matmul(A_mat.T, B_mat)
                X_mat = np.linalg.solve(AtA, AtB)

                if iter_num == 0:
                    vals, vecs = np.linalg.eig(AtA)
                    eigen_vec = vecs.copy()
                    for i in range(6):
                        if vals[i] < 100:
                            print("Warning: Degenerate!")
                            is_degenerate = True
                            eigen_vec[:, i] = np.zeros(6)
                        else:
                            break
                    P_mat = np.matmul(np.linalg.inv(vecs), eigen_vec)
                
                if is_degenerate:
                    X_mat = np.matmul(P_mat, X_mat)
                
                self.transform += np.squeeze(X_mat)
                self.transform_convert()

                delta_r = np.linalg.norm(np.rad2deg(X_mat[:3]))
                delta_t = np.linalg.norm(X_mat[3:] * 100)
                # print("{} frame, {} iter, [{},{},{}] delta translation".format(self.frame_count, iter_num, self.transform[3], self.transform[4], self.transform[5]))
                if delta_r < 0.05 and delta_t < 0.05:
                    print("Mapping converged.")
                    break

            self.transform_update()
            print("Frame: {}, Transform after mapping: {}".format(self.frame_count, self.transform))
        else:
            print("Few corner and edges in map.")

        new_points = []
        for i in range(corner_last_ds.shape[0]):
            point_sel = self.point_associate_to_map(corner_last_ds[i, :3])
            cube_i = int((point_sel[0] + 25.0) / 50.0) + self.cloud_center_width
            cube_j = int((point_sel[1] + 25.0) / 50.0) + self.cloud_center_height
            cube_k = int((point_sel[2] + 25.0) / 50.0) + self.cloud_center_depth

            if point_sel[0] + 25 < 0:
                cube_i -= 1
            if point_sel[1] + 25 < 0:
                cube_j -= 1
            if point_sel[2] + 25 < 0:
                cube_k -= 1
            
            if cube_i >=0 and cube_i < self.CLOUD_WIDTH and cube_j >= 0 and cube_j < self.CLOUD_HEIGHT and cube_k >= 0 and cube_k < self.CLOUD_DEPTH:
                cube_ind = cube_i + cube_j * self.CLOUD_WIDTH + cube_k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                self.cloud_corner_array[cube_ind].append(point_sel)
                new_points.append(point_sel)

        for i in range(surf_last_ds.shape[0]):
            point_sel = self.point_associate_to_map(surf_last_ds[i, :3])

            cube_i = int((point_sel[0] + 25.0) / 50.0) + self.cloud_center_width
            cube_j = int((point_sel[1] + 25.0) / 50.0) + self.cloud_center_height
            cube_k = int((point_sel[2] + 25.0) / 50.0) + self.cloud_center_depth

            if point_sel[0] + 25 < 0:
                cube_i -= 1
            if point_sel[1] + 25 < 0:
                cube_j -= 1
            if point_sel[2] + 25 < 0:
                cube_k -= 1
            
            if cube_i >=0 and cube_i < self.CLOUD_WIDTH and cube_j >= 0 and cube_j < self.CLOUD_HEIGHT and cube_k >= 0 and cube_k < self.CLOUD_DEPTH:
                cube_ind = cube_i + cube_j * self.CLOUD_WIDTH + cube_k * self.CLOUD_WIDTH * self.CLOUD_HEIGHT
                self.cloud_surf_array[cube_ind].append(point_sel)
                new_points.append(point_sel)

        for i in range(valid_cloud_num):
            ind = self.valid_index[i]
            if len(self.cloud_corner_array[ind]) > 0:
                _, ds_corner = downsample_filter(np.vstack(self.cloud_corner_array[ind]), 0.4)
                self.cloud_corner_array[ind] = cloud_to_list(ds_corner)

            if len(self.cloud_surf_array[ind]) > 0:
                _, ds_surf = downsample_filter(np.vstack(self.cloud_surf_array[ind]), 0.8)
                self.cloud_surf_array[ind] = cloud_to_list(ds_surf)
        
        if self.frame_count % 20 == 0:
            map_pts = []
            for i in range(self.CUBE_NUM):
                map_pts += self.cloud_surf_array[i]
                map_pts += self.cloud_corner_array[i]
            map_pts = np.vstack(map_pts)
            np.savetxt('Mapped_frame_' + str(self.frame_count) + '.txt', map_pts, fmt='%.8f')

        self.frame_count += 1
        return self.trans_w_curr









        


        