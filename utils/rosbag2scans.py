import rosbag
import sys, os
import sensor_msgs.point_cloud2
import numpy as np

def point_list_to_cloud(pts_list):
    cloud = []
    for pt in pts_list:
        pt_np = np.array([pt.x, pt.y, pt.z, pt.intensity, pt.ring])
        cloud.append(pt_np)
    cloud = np.vstack(cloud)
    return cloud

if __name__=="__main__":
    bag_name = sys.argv[1]
    output_dir = sys.argv[2]
    bag = rosbag.Bag(bag_name)
    counter = 0
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        cloud = sensor_msgs.point_cloud2.read_points_list(msg)
        cloud = point_list_to_cloud(cloud)
        np.save(os.path.join(output_dir, str(counter) + '.npy'), cloud)
        counter += 1