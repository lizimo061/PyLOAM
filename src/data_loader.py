import os
import sys
from pypcd import pypcd
import numpy as np


class DataLoader:
    def __init__(self, path, name=None):
        self.name = name
        self.path = path
        self.file_list = os.listdir(path)
        self.scan_num = len(self.file_list)
        self.sort_files()

    def __len__(self):
        return self.scan_num

    def __getitem__(self, index):
        return self.get_pc(index)

    def get_pc(self, index):
        pass

    def sort_files(self):
        pass


class PCDLoader(DataLoader):
    def get_pc(self, index):
        if index < self.scan_num:
            pcd = pypcd.PointCloud.from_path(os.path.join(self.path, self.file_list[index]))
            raw_data = pcd.pc_data
            return raw_data
        else:
            print("Access out of range!")
    
    def sort_files(self):
        self.file_list.sort(key=lambda file: int(file[:-4]))

class KittiLoader(DataLoader):
    def get_pc(self, index):
        if index < self.scan_num:
            scan = np.fromfile(os.path.join(self.path, self.file_list[index]), dtype=np.float32).reshape(-1,4)
            return scan
        else:
            print("Access out of range!")

    def sort_files(self):
        self.file_list.sort(key=lambda file: int(file[:-4]))