import numpy as np
import open3d as o3d
from utils import *

class Mapper:
    def __init__(self, config=None):
        self.rot_wmap_wodom = np.eye(3)
        self.trans_wmap_wodom = np.zeros((3,1))
        self.frame_count = 0
        # Parameters
        self.CLOUD_WIDTH = 21
        self.CLOUD_HEIGHT = 21
        self.CLOUD_DEPTH = 11
        self.CORNER_VOXEL_SIZE = 0.2
        self.SURF_VOXEL_SIZE = 0.4
        self.MAP_VOXEL_SIZE = 0.6
        self.CUBE_NUM = self.CLOUD_DEPTH * self.CLOUD_HEIGHT * self.CLOUD_WIDTH

        self.cloud_center_width = int(self.CLOUD_WIDTH/2)
        self.cloud_center_height = int(self.CLOUD_HEIGHT/2)
        self.cloud_center_depth = int(self.CLOUD_DEPTH/2)
        self.valid_index = [-1] * 125
        self.surround_index = [-1] * 125

        self.cloud_corner_array = None
        self.cloud_surf_array = None

    def transform_associate_to_map(self, rot_wodom_curr, trans_wodom_curr)
        rot_w_curr = np.matmul(self.rot_wmap_wodom, rot_wodom_curr)
        trans_w_curr = np.matmul(self.rot_wmap_wodom, trans_wodom_curr) + self.trans_wmap_wodom
        return rot_w_curr, trans_w_curr

    def map_frame(self, odom, corner_last, surf_last):
        rot_w_curr, trans_w_curr = self.transform_associate_to_map(rot_wodom_curr, trans_wodom_curr)

        cube_center_i = int((trans_w_curr[0] + 25.0) / 50.0) + self.cloud_center_width
        cube_center_j = int((trans_w_curr[1] + 25.0) / 50.0) + self.cloud_center_height
        cube_center_k = int((trans_w_curr[2] + 25.0) / 50.0) + self.cloud_center_depth

        if trans_w_curr[0] + 25.0 < 0:
            cube_center_i -= 1
        if trans_w_curr[1] + 25.0 < 0:
            cube_center_j -= 1
        if trans_w_curr[2] + 25.0 < 0:
            cube_center_k -= 1
        
        while cube_center_i < 3:
            for j in range(self.CLOUD_HEIGHT):
                for k in range(self.CLOUD_DEPTH):
                    pass
                    # TODO: Move the cube
            
            cube_center_i += 1
            self.cloud_center_width += 1
        
        while cube_center_i >= self.CLOUD_WIDTH - 3:
            for j in range(self.CLOUD_HEIGHT):
                for k in range(self.CLOUD_DEPTH):
                    pass
            
            cube_center_i -= 1
            self.cloud_center_width -= 1

        while cube_center_j < 3:
            for i in range(self.CLOUD_WIDTH):
                for k in range(self.CLOUD_DEPTH):
                    pass

            cube_center_j += 1
            self.cloud_center_height += 1
        
        while cube_center_j >= self.CLOUD_HEIGHT - 3:
            for i in range(self.CLOUD_WIDTH):
                for k in range(self.CLOUD_DEPTH):
                    pass
            
            cube_center_j -= 1
            self.cloud_center_height -= 1

        while cube_center_k < 3:
            for i in range(self.CLOUD_WIDTH):
                for j in range(self.CLOUD_HEIGHT):
                    pass
            
            cube_center_k += 1
            self.cloud_center_depth += 1
        
        while cube_center_k >= self.CLOUD_DEPTH - 3:
            for i in range(self.CLOUD_WIDTH):
                for j in range(self.CLOUD_HEIGHT):
                    pass
            
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
                        self.surround_index[surround_index] = i + j * self.CLOUD_WIDTH + k * self.CLOUD_HEIGHT * self.CLOUD_WIDTH
                        surround_index += 1
        
        map_corner_list = []
        map_surf_list = []
        for i in range(valid_cloud_num):
            map_corner_list.append(self.cloud_corner_array[self.valid_index[i]])
            map_surf_list.append(self.cloud_surf_array[self.valid_index[i]])
        
        if len(map_corner_list) > 0:
            corner_from_map = np.vstack(map_corner_list)

        if len(map_surf_list) > 0:
            surf_from_map = np.vstack(map_surf_list)
        
        _, corner_last_ds = downsample_filter(corner_last, self.CORNER_VOXEL_SIZE)
        _, surf_last_ds = downsample_filter(surf_last, self.SURF_VOXEL_SIZE)

        if len(map_corner_list) > 0 and len(map_surf_list) > 0 and corner_from_map.shape[0] > 10 and surf_from_map.shape[0] > 100:
            corner_map_tree = o3d.geometry.KDTreeFlann(np.transpose(corner_from_map[:, :3]))
            surf_map_tree = o3d.geometry.KDTreeFlann(np.transpose(surf_from_map[:, :3]))

            for iter_num in range(10):
                #TODO: Find matches and get transformation



        


        