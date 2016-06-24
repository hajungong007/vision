#!/usr/bin/env python
import roslib
#roslib.load_manifest('python_object_scanner')
import sys
import copy
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image as ImageMsg
from deep_cnn_object_detection.srv import *
from datetime import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError

import yaml

import argparse
import os

from datetime import datetime
    
class image_subscriber:
    def __init__(self, object_name):
	self.object_name = object_name
	self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()        
        #rospy.wait_for_service('getImage')
        self.getImage_client = rospy.ServiceProxy('image_preparator/getImage', getImage)
        self.UpdateTFs_client = rospy.ServiceProxy('image_preparator/UpdateTFs', UpdateTFs)
        
	
	dir_path = '/home/msdu/datasets/images/' + self.object_name
	if not os.path.exists(dir_path):
	    os.makedirs(dir_path)
	pass

    def get_image(self):
	res = self.getImage_client()
	data = res.msg
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # bgr or rgb it is no matter, it will be in bgr in both cases
        except CvBridgeError as e:
            print e  
    def save_image(self, image_num):
	image_path = '/home/msdu/datasets/images/' + self.object_name + '/' + str(image_num) + '.jpg'
	cv2.imwrite(image_path, self.image)
	pass

class robot_mover:
    def __init__(self, image_sub):
	self.rospack = rospkg.RosPack()
	self.rotation_angle = 10
	self.robot = moveit_commander.RobotCommander()
	self.scene = moveit_commander.PlanningSceneInterface()
	self.left_group = moveit_commander.MoveGroupCommander("left_arm")
	self.right_group = moveit_commander.MoveGroupCommander("right_arm")
	self.deg_to_rad = 3.1415926 / 180;	
	self.image_sub = image_sub
	self.left_start_point = []
	self.right_points = []
	self.parse_conf()	
        pass    
    def parse_conf(self):
	
	# TODO: Parse points for both arms left and right	
	# create start_points and observe_points variables for left and right arms
	
	prefix = self.rospack.get_path('deep_cnn_object_detection')
	conf_path = prefix + '/conf/points.yaml'
	with open(conf_path, 'r') as stream:
	    conf = yaml.load(stream)
	    for joint, value in conf['left_start_point'].iteritems():
		self.left_start_point.append(value)
	    del conf['left_start_point']
	    for num, point in conf.iteritems():
		point_as_list = []
		for joint, value in point.iteritems():
		    point_as_list.append(value)
		    pass
		self.right_points.append(point_as_list)
		pass
	pass
    def gotoStartPoses(self):
	self.gotoStartPoseLeft()
	self.gotoStartPoseRight()
	pass    
    def gotoStartPoseLeft(self):
	left_group_variable_values = self.left_start_point
	print "Moving left arm to point: ",  left_group_variable_values
	left_group_variable_values[:] = [x * self.deg_to_rad for x in left_group_variable_values] 
	self.left_group.set_joint_value_target(left_group_variable_values)
	plan = self.left_group.plan()
	self.left_group.execute(plan, True)
	print "left_arm ready"
	pass
    def gotoStartPoseRight(self):		
	right_group_variable_values = list(self.right_points[0])
	print "Moving right arm to point: ",  right_group_variable_values
	right_group_variable_values[:] = [x * self.deg_to_rad for x in right_group_variable_values] 
	self.right_group.set_joint_value_target(right_group_variable_values)
	plan = self.right_group.plan()
	self.right_group.execute(plan, True)
	print "right_arm ready"
	pass 
    def Scan(self):
	start_scan_time = datetime.now()
	number_of_scans = 360 / self.rotation_angle
	print "Start scanning, num_scans: ", number_of_scans
	image_num = 0
	left_group_variable_values = self.left_start_point
	for point in self.right_points:
	    print "Moving right arm to point: ",  point
	    right_group_variable_values = point
	    right_group_variable_values[:] = [x * self.deg_to_rad for x in right_group_variable_values] 
	    self.right_group.set_joint_value_target(right_group_variable_values)
	    plan = self.right_group.plan()
	    self.right_group.execute(plan, True)
	    print "right_arm ready"
	    
	    self.image_sub.UpdateTFs_client()
	    
	    for scan_number in xrange(number_of_scans):	
		print "---------------- Scanning point # ", scan_number, "---------------------------"
		left_group_variable_values[5] = (180.0 - scan_number * self.rotation_angle) * self.deg_to_rad;
		self.left_group.set_joint_value_target(left_group_variable_values)
		
		print "planning"
		start_time_global = datetime.now()
		plan = self.left_group.plan()		
		end_time = datetime.now()
		delta_time = end_time - start_time_global
		print "planned: ", delta_time, "; executing"
		
		start_time = datetime.now()
		self.left_group.execute(plan, True)
		end_time = datetime.now()
		delta_time = end_time - start_time
		print "executed ", delta_time, "; getting image"
		
		#self.image_sub.UpdateTFs_client()
		start_time = datetime.now()
		self.image_sub.get_image()
		end_time = datetime.now()
		delta_time = end_time - start_time
		print "image got ", delta_time, "; saving"
		
		start_time = datetime.now()
		self.image_sub.save_image(image_num)
		
		self.left_group.set_start_state_to_current_state()
		#rospy.sleep(1)	
		
		end_time = datetime.now()
		delta_time = end_time - start_time
		print "done ", delta_time
		print "whole time: ", end_time - start_time_global
		image_num += 1
		pass
	    self.right_group.set_start_state_to_current_state()
	    #rospy.sleep(1)
	    pass
	print "Scanning ended! Time consumed: ", datetime.now() - start_scan_time 
	pass
    def RotateLeftArm(self):
	pass

def parse_args():	
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', dest='name', type=str, help='Object name', required = True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('object_scanning_node', anonymous=True)
    image_sub = image_subscriber(args.name)
    mover = robot_mover(image_sub)
    mover.gotoStartPoses()   
    mover.Scan()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    moveit_commander.roscpp_shutdown()
    print "scanning done"
    pass