import numpy as np
import open3d as o3d

class Mapper:
    def __init__(self, config=None):
        self.rot_wmap_wodom = np.eye(3)
        self.trans_wmap_wodom = np.zeros((3,1))
        self.frame_count = 0

    def transform_associate_to_map(self, rot_wodom_curr, trans_wodom_curr)
        rot_w_curr = np.matmul(self.rot_wmap_wodom, rot_wodom_curr)
        trans_w_curr = np.matmul(self.rot_wmap_wodom, trans_wodom_curr) + self.trans_wmap_wodom
        return rot_w_curr, trans_w_curr

    def add_frame(self):
        
        