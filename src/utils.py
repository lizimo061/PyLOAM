from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def get_rotation(rx, ry, rz):
    r = R.from_euler('yxz', [rz, rx, ry], degrees=False)
    return r.as_dcm()