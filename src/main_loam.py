from data_loader import NPYLoader
from laser_odometry import Odometry
from laser_mapping import Mapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path', type=str, help='Input folder of npy files')

if __name__== '__main__':
    args = parser.parse_args()

    odometry = Odometry()
    mapper = Mapper()
    loader = NPYLoader(path=args.npy_path, name='NSH indoor')

    for i in range(len(loader)):
        cloud = loader[i]
        surf_pts, corner_pts, odom = odometry.grab_frame(cloud)
        trans = mapper.map_frame(odom, corner_pts, surf_pts)