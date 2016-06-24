#!/usr/bin/env python
import roslib
#roslib.load_manifest('python_object_scanner')
import sys
import copy
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

class joint_vel_calc(object):
    def __init__(self):
	self.prev_value = 0.0
	self.value = 0.0
	rospy.init_node('joint_vel_calc', anonymous=True)
	self.pub = rospy.Publisher('/joint_velocity', Float32, queue_size=10)
	rospy.Subscriber("/joint_states", JointState, self.callback)
        pass
    def callback(self, data):
	self.value = data.position[5]
	#self.pub.publish(vel)
	pass    
    def run(self):
	r = rospy.Rate(10)
        while not rospy.is_shutdown():
	    vel = self.value - self.prev_value
	    self.prev_value = self.value
            self.pub.publish(vel)
            r.sleep()
	pass
    

if __name__ == '__main__':

    _joint_vel_calc = joint_vel_calc()    
    _joint_vel_calc.run()

    pass