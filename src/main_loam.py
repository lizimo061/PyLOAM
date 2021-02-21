from data_loader import NPYLoader
from laser_odometry import Odometry
from laser_mapping import Mapper
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path', type=str, help='Input folder of npy files')
parser.add_argument('--config_path', type=str, help='Configuration file')

if __name__== '__main__':
    args = parser.parse_args()
    loader = NPYLoader(path=args.npy_path, name='NSH indoor')

    config = None
    if args.config_path is not None:
        with open(args.config_path) as config_file:
            config_data = config_file.read()
            config = json.loads(config_data)
    
    odometry = Odometry(config=config)
    mapper = Mapper(config=config)

    for i in range(len(loader)):
        cloud = loader[i]
        surf_pts, corner_pts, odom = odometry.grab_frame(cloud)
        trans = mapper.map_frame(odom, corner_pts, surf_pts)