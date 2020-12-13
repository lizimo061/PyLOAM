from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import open3d as o3d

def get_rotation(rx, ry, rz):
    r = R.from_euler('yxz', [ry, rx, rz], degrees=False)
    return r.as_dcm()

def get_euler_angles(rot_mat):
    r = R.from_dcm(rot_mat)
    deg = r.as_euler('yxz', degrees=False)
    return deg[1], deg[0], deg[2]

def downsample_filter(cloud, voxel_size):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3])
    max_bound = o3d_cloud.get_max_bound() + voxel_size * 0.5
    min_bound = o3d_cloud.get_min_bound() - voxel_size * 0.5
    out = o3d_cloud.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, False)
    index_ds = [cubic_index[0] for cubic_index in out[2]]
    cloud_ds = cloud[index_ds, :]
    return index_ds, cloud_ds

def get_mean_cov(cloud):
    """
        Assuming N by 3 input
    """
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3])
    mean_cov = o3d_cloud.compute_mean_and_covariance()
    return mean_cov[0], mean_cov[1]

def cloud_to_list(cloud):
    return [cloud[i, :3].reshape(1,3) for i in range(cloud.shape[0])]