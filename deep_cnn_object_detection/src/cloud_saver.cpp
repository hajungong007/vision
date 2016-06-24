
#include <deep_cnn_object_detection/cloud_saver.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud;

void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& pc_msg)
{
    ROS_INFO_STREAM("Cloud received");
    input_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*pc_msg, *input_cloud);
    ros::Duration(0.1).sleep();
}
tf::StampedTransform GetTransform(std::string parent, std::string child)
{
    tf::TransformListener listener;
    tf::StampedTransform out;
    try
    {
	if (listener.waitForTransform(parent, child, ros::Time(0), ros::Duration(5)))
	{
	    listener.lookupTransform(parent, child, ros::Time(0), out); 
	    return out;
	}
	else 
	{
	    ROS_INFO_STREAM("Reading transform " << parent << "->" << child << " failed");
	}
    }
    catch (tf::TransformException ex)
    {
	ROS_ERROR("%s",ex.what());
	ros::Duration(1).sleep();
    }
}
bool SaveImage(deep_cnn_object_detection::SaveCloud::Request &req,
	       deep_cnn_object_detection::SaveCloud::Response &res)
{
//     tf::StampedTransform transform_world_camera = GetTransform("/world", "/camera_rgb_optical_frame");
//     Eigen::Affine3d TransformEigen_world_camera;
//     tf::transformTFToEigen(transform_world_camera, TransformEigen_world_camera);    
//     pcl::transformPointCloud(*input_cloud, *output_cloud, TransformEigen_world_camera);
    std::string file_path = "/home/msdu/catkin_ws/src/vision/deep_cnn_object_detection/clouds/" + std::to_string(req.angle) + ".pcd";
    pcl::io::savePCDFileASCII(file_path, *input_cloud);
    return true;
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_two_ints_server");
    ros::NodeHandle nh_;
    std::string subs_cloud_topic_ = "/camera/depth_registered/points";
    ros::Subscriber pc_subs_ = nh_.subscribe(subs_cloud_topic_.c_str(),1, PointCloudCallback);

    ros::ServiceServer getImage_service = nh_.advertiseService("cloud_saver/SaveCloud", &SaveImage);      
    
    

    while (ros::ok())
    {
	ros::spinOnce();
	ros::Duration(0.5).sleep();
    }

    return 0;
}