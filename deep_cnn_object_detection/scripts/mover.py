#!/usr/bin/env python
import roslib
#roslib.load_manifest('save_clouds')
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

class robot_mover:
    def __init__(self):
	self.rospack = rospkg.RosPack()
	self.rotation_angle = 20.0
	self.robot = moveit_commander.RobotCommander()
	self.scene = moveit_commander.PlanningSceneInterface()
	self.left_group = moveit_commander.MoveGroupCommander("left_arm")
	self.right_group = moveit_commander.MoveGroupCommander("right_arm")
	self.deg_to_rad = 3.1415926 / 180;
        pass    

    def Scan(self):
	

	left_group_variable_values = self.left_group.get_current_joint_values()
	left_group_variable_values[5] = 0.0
	sign = 1
	for i in xrange(1000):		
	    left_group_variable_values[5] += 0.2 * sign
	    self.left_group.set_joint_value_target(left_group_variable_values)
	    print "planning to: ", left_group_variable_values[5]
	    plan = self.left_group.plan()
	    print "planned, executing"
	    self.left_group.execute(plan, True)
	    prev_point = 0.0
	    for point in plan.joint_trajectory.points:
		print "-------- NEXT POINT ---------"
		print "position: ", point.positions[5]
		print "velocity: ", point.positions[5] - prev_point
		prev_point = point.positions[5]	    

	    print "executed"
	    sign *= -1
	    self.left_group.set_start_state_to_current_state()
	    rospy.sleep(1)	    
	    pass
	print "Scanning ended!"
	pass
    def RotateLeftArm(self):
	pass

if __name__ == '__main__':   
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('object_scanning_node', anonymous=True)
    mover = robot_mover()
    mover.Scan()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    moveit_commander.roscpp_shutdown()
    print "scanning done"
    pass