import numpy as np
import os

def load_data():
    hists_path = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/hists.txt"  
    hists = np.loadtxt(open(hists_path))
    
    rolls_path = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/rolls.txt"  
    rolls = np.loadtxt(open(rolls_path))
    
    poses_path = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/poses.txt"  
    poses = np.loadtxt(open(poses_path))
    
    return hists, rolls, poses
