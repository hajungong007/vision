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
                                            sensor, res=SXGA_RES, fps=FPS_30,
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
    # openni&freenect requires '/camera/rgb/image_rect_color', openni2 just need '/camera/rgb/image_raw'
    if sensor in ['kinect']: 
        rgb_image_sub = ecto_ros.ecto_sensor_msgs.Subscriber_Image("rgb_image_sub", topic_name='/camera/rgb/image_color')
    elif sensor in ['xtion']: 
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
    """            
    rgb_display = highgui.imshow(name='rgb')
    depth_display = highgui.imshow(name='depth')
    graph += [image2cv[:] >> rgb_display[:],
              depth2cv[:] >> depth_display[:],
            ]
            
    #   Create pointcloud viewer
    viewer = CloudViewer("viewer", window_name="Clouds!")    
    graph += [ msg2cloud[:] >> viewer[:] ]
    """   
#   Process pointcloud

    # Cut off some parts
    
    cut_x = PassThrough(filter_field_name="x", 
                        filter_limit_min=-0.2,
                        filter_limit_max=0.2)
                        
    cut_y = PassThrough(filter_field_name="y", 
                        filter_limit_min=-0.5,
                        filter_limit_max=0.5)
    cut_z = PassThrough(filter_field_name="z",    
                        filter_limit_min=-0.0,
                        filter_limit_max=1.0)
                        
    graph += [ msg2cloud[:] >> cut_x[:],
               cut_x[:] >> cut_y[:],
               cut_y[:] >> cut_z[:] ]
               
    # Voxel filter
    
    voxel_grid = VoxelGrid("voxel_grid", leaf_size=0.002)
    
    graph += [ cut_z[:] >> voxel_grid[:] ]
               
    # Find plane and delete it
    
    normals = NormalEstimation("normals", k_search=0, radius_search=0.02)
    
    graph += [ voxel_grid[:] >> normals[:] ]
    
    plane_finder = SACSegmentationFromNormals("planar_segmentation",
                                                 model_type=SACMODEL_NORMAL_PLANE,
                                                 eps_angle=0.09, distance_threshold=0.1)
    
    plane_deleter = ExtractIndices(negative=True)
    
    graph += [ voxel_grid[:] >> plane_finder['input'],
               normals[:] >> plane_finder['normals'] ]
    graph += [ voxel_grid[:]  >> plane_deleter['input'],
               plane_finder['inliers'] >> plane_deleter['indices'] ]
#   Create pointcloud viewer
    viewer = CloudViewer("viewer", window_name="Clouds!")    
    graph += [ plane_deleter[:] >> viewer[:] ]
    
#   Compute vfh
    
    vfh_signature = VFHEstimation(radius_search=0.01)
    
    graph += [ plane_deleter[:] >>  vfh_signature['input'],
               normals[:] >> vfh_signature['normals'] ]                                                          
    
    # convert the image to grayscale                                                                      
    rgb2gray = imgproc.cvtColor('rgb -> gray', flag=imgproc.Conversion.RGB2GRAY)
    graph += [image2cv[:]  >> rgb2gray[:] ]
                
    # get a pose use the dot pattern: there might be a scale ambiguity as this is 3d only           
    poser = OpposingDotPoseEstimator(rows=5, cols=3,
                                     pattern_type=calib.ASYMMETRIC_CIRCLES_GRID,
                                     square_size=0.04, debug=True)
    graph += [ image2cv[:] >> poser['color_image'],
               rgb_info2cv['K'] >> poser['K_image'],
               rgb2gray[:] >> poser['image'] ]         #poser gives us R&T from camera to dot pattern
    
    
    points3d = calib.DepthTo3d()
    
    graph += [ depth_info2cv['K'] >> points3d['K'],
               depth2cv['image'] >> points3d['depth']
             ]
    
    plane_est = PlaneFinder(min_size=10000)
    compute_normals = ComputeNormals()
    # Convert K if the resolution is different (the camera should output that)
    
    graph += [ # find the normals
                depth_info2cv['K'] >> compute_normals['K'],
                points3d['points3d'] >> compute_normals['points3d'],
                # find the planes
                compute_normals['normals'] >> plane_est['normals'],
                depth_info2cv['K'] >> plane_est['K'],
                points3d['points3d'] >> plane_est['points3d'] ]
    

    
    pose_filter = object_recognition_capture.ecto_cells.capture.PlaneFilter();

    # make sure the pose is centered at the origin of the plane
    graph += [ depth_info2cv['K'] >> pose_filter['K_depth'],
               poser['R', 'T'] >> pose_filter['R', 'T'],
               plane_est['planes', 'masks'] >> pose_filter['planes', 'masks'] ]
               
    delta_pose = ecto.If('delta R|T', cell=object_recognition_capture.DeltaRT(angle_thresh=angle_thresh,
                                                          n_desired=n_desired))
                                                          
    poseMsg = RT2PoseStamped(frame_id='/camera_rgb_optical_frame')

    graph += [ pose_filter['R', 'T', 'found'] >> delta_pose['R', 'T', 'found'],
               pose_filter['R', 'T'] >> poseMsg['R', 'T'] ]
    
    trainer_ = Trainer()
    
    graph += [ pose_filter['R', 'T'] >> trainer_['R', 'T'],
               vfh_signature['output'] >> trainer_['features'],
               delta_pose['last'] >> trainer_['last'],
               delta_pose['novel'] >> trainer_['novel'],
               plane_deleter[:] >> trainer_['input'] ]
 
    novel = delta_pose['novel']

    cloud2msg = ecto_pcl_ros.PointCloud2Message("cloud2msg")

    graph += [ plane_deleter[:] >> cloud2msg[:] ]   
    
    baggers = dict(points=PointCloudBagger(topic_name='/camera/depth_registered/points'),
                   pose=PoseBagger(topic_name='/camera/pose')
                   )
                   
    bagwriter = ecto.If('Bag Writer if R|T', cell=ecto_ros.BagWriter(baggers=baggers, bag=bag_name))
    pcd_writer = ecto.If('msd pcd writer', cell = PCDWriter())
    
    graph += [    cloud2msg[:] >> bagwriter['points'],
                  poseMsg['pose'] >> bagwriter['pose']
                  ]
    graph += [ plane_deleter['output'] >> pcd_writer['input'] ]
    
    delta_pose.inputs.__test__ = True
    pcd_writer.inputs.__test__ = True
    
    graph += [novel >> (bagwriter['__test__'])]  
    graph += [novel >> (pcd_writer['__test__'])]
    
    rt2pose = RT2PoseStamped(frame_id = "camera_rgb_optical_frame")
    rt2pose_filtered = RT2PoseStamped(frame_id = "camera_rgb_optical_frame")      
    posePub = PosePublisher("pose_pub", topic_name='pattern_pose')
    posePub_filtered = PosePublisher("pose_pub_filtered", topic_name='pattern_pose_filtered')
    graph += [ poser['R', 'T'] >> rt2pose['R', 'T'],
               rt2pose['pose'] >> posePub['input'],
               pose_filter['R', 'T'] >> rt2pose_filtered['R', 'T'],
               rt2pose_filtered['pose'] >> posePub_filtered['input'] ]
   
    plasm = ecto.Plasm()
    plasm.connect(graph)
    return (plasm, segmentation_cell) # return segmentation for tuning of parameters.

