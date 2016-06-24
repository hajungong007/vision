#!/usr/bin/env python
"""
Module defining the msd-pcl detector to find objects in a scene
"""

from object_recognition_core.pipelines.detection import DetectorBase
#import object_recognition_msd_pcl.ecto_cells.ecto_msd_pcl as ecto_msd_pcl
from object_recognition_msd_pcl.ecto_cells.ecto_msd_pcl import Detector as Detector

from ecto_image_pipeline.base import CameraModelToCv
from ecto_image_pipeline.io.source import create_source
from ecto_opencv import highgui, calib, imgproc
from ecto_opencv.rgbd import ComputeNormals, PlaneFinder
#from ecto_openni import SXGA_RES, FPS_30
from ecto_ros import Cv2CameraInfo, Mat2Image, RT2PoseStamped
from ecto_ros.ecto_geometry_msgs import Bagger_PoseStamped as PoseBagger, Publisher_PoseStamped as PosePublisher
from ecto_ros.ecto_sensor_msgs import Bagger_Image as ImageBagger, Bagger_CameraInfo as CameraInfoBagger, Bagger_PointCloud2 as PointCloudBagger
#from object_recognition_capture.fiducial_pose_est import OpposingDotPoseEstimator
from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward

from ecto_pcl import *
import ecto
import ecto_ros
import ecto_pcl_ros, ecto_ros.ecto_sensor_msgs
import math
#import object_recognition_capture
import time
import sys
########################################################################################################################

class MsdPclDetector(ecto.BlackBox, DetectorBase):

    def __init__(self, *args, **kwargs):
#        ecto_msd_pcl.Detector.__init__(self, *args, **kwargs)
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @classmethod
    def declare_cells(cls, _p):
        return {
                'detector_' : Detector(#search_json_params=_p['search'],
                            json_db=_p['json_db'],
                            json_object_ids=_p['json_object_ids'],
                            #quite = False
                            )
                }

    @staticmethod
    def declare_forwards(p):
        p = { }

        i = { }

        o = {'detector_': [Forward('pose_results')]
            }

        return (p,i,o) 

    @classmethod
    def declare_direct_params(self, p):
        p.declare('json_db', 'The DB to get data from as a JSON string', '{}')
        p.declare('search', 'The search parameters as a JSON string', '{}')
        p.declare('json_object_ids', 'The ids of the objects to find as a JSON list or the keyword "all".', 'all')  
        p.declare('sensor', 'Type of a sensor', '')  
        
    def configure(self, p, _i, _o):
        #if p.sensor=='kinect2':
            #self.cloud_sub = ecto_ros.ecto_sensor_msgs.Subscriber_PointCloud2("cloud_sub", topic_name='/kinect2_head/depth_lowres/points') 
        #else:
            #self.cloud_sub = ecto_ros.ecto_sensor_msgs.Subscriber_PointCloud2("cloud_sub", topic_name='/camera/depth_registered/points') 
        self.cloud_sub = ecto_ros.ecto_sensor_msgs.Subscriber_PointCloud2("cloud_sub", topic_name='/selected_cloud')
        #self.cloud_sub = ecto_ros.ecto_sensor_msgs.Subscriber_PointCloud2("cloud_sub", topic_name='/camera/depth_registered/points')
        self.msg2cloud = ecto_pcl_ros.Message2PointCloud("msg2cloud", format=XYZRGB)
        self.cut_x = PassThrough(filter_field_name="x", 
                        filter_limit_min=-0.5,
                        filter_limit_max=0.5)
        self.cut_y = PassThrough(filter_field_name="y", 
                        filter_limit_min=-0.5,
                        filter_limit_max=0.5)
        self.cut_z = PassThrough(filter_field_name="z",    
                        filter_limit_min=0.5,
                        filter_limit_max=1.3)
        self.voxel_grid = VoxelGrid("voxel_grid", leaf_size=0.001)
        self.viewer = CloudViewer("viewer", window_name="PCD Viewer", )
                
             
    def connections(self, _p):
        graph = []
        graph += [ self.cloud_sub['output'] >> self.msg2cloud[:]
        ]
        #graph += [ self.msg2cloud[:] >> self.cut_x[:],
        #            self.cut_x[:] >> self.cut_y[:],
        #            self.cut_y[:] >> self.cut_z[:] ]
        #graph += [ self.cut_z[:] >> self.voxel_grid[:] ]
        graph += [ self.msg2cloud[:] >> self.voxel_grid[:] ]        
        graph += [ self.voxel_grid[:] >> self.detector_['input'],
#                   self.voxel_grid[:] >> self.viewer[:]
                    ]
            
        return graph
########################################################################################################################

