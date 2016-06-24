"""
Module defining common tools for object capture
"""
from ecto_image_pipeline.base import CameraModelToCv
from ecto_image_pipeline.io.source import create_source
from ecto_opencv import highgui, calib, imgproc
from ecto_opencv.rgbd import ComputeNormals, PlaneFinder
from ecto_openni import SXGA_RES, FPS_30
from ecto_ros import Cv2CameraInfo, Mat2Image, RT2PoseStamped
from ecto_ros.ecto_geometry_msgs import Bagger_PoseStamped as PoseBagger, Publisher_PoseStamped as PosePublisher
from ecto_ros.ecto_sensor_msgs import Bagger_Image as ImageBagger, Bagger_CameraInfo as CameraInfoBagger, Bagger_PointCloud2 as PointCloudBagger
from object_recognition_capture.fiducial_pose_est import OpposingDotPoseEstimator
from ecto_pcl import *
import ecto
import ecto_ros
import ecto_pcl_ros, ecto_ros.ecto_sensor_msgs
import math
import object_recognition_capture
import time
import sys

from object_recognition_msd_pcl.ecto_cells.ecto_msd_pcl import Trainer


def create_capture_plasm(bag_name, angle_thresh, segmentation_cell, n_desired,
                                            orb_template='', res=SXGA_RES, fps=FPS_30,
                                            orb_matches=False,
                                            preview=False, use_turn_table=True):
    '''
    Creates a plasm that will capture openni data into a bag, using a dot pattern to sparsify views.
    
    @param bag_name: A filename for the bag, will write to this file.
    @param angle_thresh: The angle threshhold in radians to sparsify the views with.  
    '''
    graph = []
    
#   Create ros node, subscribing to openni topics
    argv = sys.argv[:]
    ecto_ros.strip_ros_args(argv)
    ecto_ros.init(argv, "object_capture_msd_pcl", False)
    cloud_sub = ecto_ros.ecto_sensor_msgs.Subscriber_PointCloud2("cloud_sub", topic_name='/camera/depth_registered/points') 
    rgb_image_sub = ecto_ros.ecto_sensor_msgs.Subscriber_Image("rgb_image_sub", topic_name='/camera/rgb/image_raw')
    depth_image_sub = ecto_ros.ecto_sensor_msgs.Subscriber_Image("depth_image_sub", topic_name='/camera/depth_registered/image_raw')  
    rgb_camera_info_sub = ecto_ros.ecto_sensor_msgs.Subscriber_CameraInfo("rgb_camera_info_sub", topic_name='/camera/rgb/camera_info')
    depth_camera_info_sub = ecto_ros.ecto_sensor_msgs.Subscriber_CameraInfo("depth_camera_info_sub", topic_name='/camera/depth_registered/camera_info')
        
    
#   Converte ros topics to cv types
    msg2cloud = ecto_pcl_ros.Message2PointCloud("msg2cloud", format=XYZRGB)
    image2cv = ecto_ros.Image2Mat()
    depth2cv = ecto_ros.Image2Mat()
    rgb_info2cv = ecto_ros.CameraInfo2Cv()
    depth_info2cv = ecto_ros.CameraInfo2Cv()


    graph += [ cloud_sub['output'] >> msg2cloud[:],
               rgb_image_sub[:] >> image2cv[:],
               depth_image_sub[:] >> depth2cv[:],
               rgb_camera_info_sub[:] >> rgb_info2cv[:],
               depth_camera_info_sub[:] >> depth_info2cv[:]
                ]
                                                        
    
     #display the mask

    display = highgui.imshow(name='Poses')
    graph += [image2cv[:] >> display[:],
            ]
    
    # convert the image to grayscale                                                                      
#    rgb2gray = imgproc.cvtColor('rgb -> gray', flag=imgproc.Conversion.RGB2GRAY)
#    graph += [image2cv[:]  >> rgb2gray[:] ]
    
        # get a pose use the dot pattern: there might be a scale ambiguity as this is 3d only           
    poser = OpposingDotPoseEstimator(rows=5, cols=3,
                                     pattern_type=calib.ASYMMETRIC_CIRCLES_GRID,
                                     square_size=0.04, debug=True)
    graph += [ image2cv[:] >> poser['color_image'],
               rgb_info2cv['K'] >> poser['K_image'],
               image2cv[:] >> poser['image'] ] 
    
    plasm = ecto.Plasm()
    plasm.connect(graph)
    return (plasm, segmentation_cell) # return segmentation for tuning of parameters.

