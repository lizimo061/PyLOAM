from data_loader import KittiLoader
from laser_odometry import Odometry
from laser_mapping import Mapper
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--kitti_path', type=str, help='Input folder of KITTI .bin files')
parser.add_argument('--config_path', type=str, help='Configuration file')

if __name__== '__main__':
    args = parser.parse_args()
    loader = KittiLoader(path=args.kitti_path, name='Kitti dataset')

    config = None
    if args.config_path is not None:
        with open(args.config_path) as config_file:
            config_data = config_file.read()
            config = json.loads(config_data)
    
    odometry = Odometry(config=config)
    mapper = Mapper(config=config)
    skip_frame = 5
    res_mapped = []
    res_odom = []

    for i in range(len(loader)):
        cloud = loader[i]
        surf_pts, corner_pts, odom = odometry.grab_frame(cloud)
        res_odom.append(odom[0:3, 3].reshape(3))
        if i % skip_frame == 0:
            trans = mapper.map_frame(odom, corner_pts, surf_pts)
            if trans is not None:
                res_mapped.append(trans.reshape(3))

    res = np.vstack(res_mapped)
    np.savetxt("mapped.txt", res, fmt='%.8f')

    res = np.vstack(res_odom)
    np.savetxt("odom.txt", res, fmt='%.8f')