from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import open3d as o3d

def get_rotation(rx, ry, rz):
    r = R.from_euler('yxz', [ry, rx, rz], degrees=False)
    return r.as_dcm()

def downsample_filter(cloud, voxel_size):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3])
    max_bound = o3d_cloud.get_max_bound() + voxel_size * 0.5
    min_bound = o3d_cloud.get_min_bound() - voxel_size * 0.5
    out = o3d_cloud.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, False)
    index_ds = [cubic_index[0] for cubic_index in out[2]]
    cloud_ds = cloud[index_ds, :]
    return index_ds, cloud_ds