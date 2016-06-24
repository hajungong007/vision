#include <ros/ros.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <object_recognition_msgs/RecognizedObjectArray.h>
#include <object_recognition_ros/object_info_cache.h>
#include <object_recognition_msgs/ObjectType.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

ros::Publisher pose_pub;

void ORKCallback(const object_recognition_msgs::RecognizedObjectArray::ConstPtr& msg)
{
    if (msg->objects.size() > 0)
    {
	geometry_msgs::PoseStamped pose_msg;
	pose_msg.pose = msg->objects[0].pose.pose.pose;
	pose_msg.header.frame_id = msg->objects[0].header.frame_id;
	pose_pub.publish(pose_msg);
    }
}
int main( int argc, char **argv )
{
    ros::init(argc, argv, "ork_to_pose_converter_node");
    ros::NodeHandle nh;
    
    ros::Subscriber rec_obj_array_sub = nh.subscribe("/recognized_object_array_", 1000, &ORKCallback);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/recognized_object_pose", 1);
    
    while (ros::ok())
    {
	ros::spinOnce();
	ros::Duration(0.1).sleep();
    }
    return 0;
}
