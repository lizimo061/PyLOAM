import numpy as np
import open3d as o3d
from feature_extract import FeatureExtract
from minisam import *

class Odometry:
    def __init__(self, config=None):
        self.config = config
        self.init = False
        self.surf_last = None
        self.corner_last = None
        self.feature_extractor = FeatureExtract()
        self.DIST_THRES = 25
        self.RING_INDEX = 4

    def grab_frame(self, cloud):
        corner_sharp, corner_less, surf_flat, surf_less = self.feature_extractor.feature_extract(cloud)
        if not self.init:
            self.init = True
            self.surf_last = surf_less
            self.corner_last = corner_less
        else:
            pass

    def get_corner_correspondences(self, corner_sharp):
        corner_last_tree = o3d.geometry.KDTreeFlann(np_o3d(self.corner_last))

        for i in range(corner_sharp.shape[0]):
            point_sel = corner_sharp[i, :3]
            _, ind, dist = corner_last_tree.search_knn_vector_3d(point_sel, 1)
            closest_ind = -1
            min_ind2 = -1
            if dist[0] < self.DIST_THRES:
                closest_ind = ind
                # TODO
    
    def get_surf_correspondences(self, surf_flat):
        pass

    @staticmethod
    def np_o3d(cloud):
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(cloud[:, :3])
        return pc_o3d


class EdgeErrorFactor(Factor):
    pass

class PlaneErrorFactor(Factor):
    pass