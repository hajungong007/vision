#include <ros/ros.h>
#include <deep_cnn_object_detection/image_preparator.h>

// bool add(beginner_tutorials::AddTwoInts::Request  &req,
//          beginner_tutorials::AddTwoInts::Response &res)
// {
//   res.sum = req.a + req.b;
//   ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
//   ROS_INFO("sending back response: [%ld]", (long int)res.sum);
//   return true;
// }
ImagePreparator::ImagePreparator()
{
    subs_cloud_topic_ = std::string("/camera/depth_registered/points");
    pc_subs_ =  nh_.subscribe(subs_cloud_topic_.c_str(),1,&ImagePreparator::PointCloudCallback, this);
    input_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    transformed_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cutted_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    object_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    object_cloud_filtered.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    blackened_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    image_publisher = nh_.advertise<sensor_msgs::Image>( "/blackened_image", 1 );
    
    transform_world_camera = GetTransform("/world", "/camera_rgb_optical_frame");
    transform_world_rotary_table = GetTransform("/world", "/rotary_table");
    tf::transformTFToEigen(transform_world_camera, TransformEigen_world_camera);
}
bool ImagePreparator::getImage(deep_cnn_object_detection::getImage::Request& req, deep_cnn_object_detection::getImage::Response& res)
{
    ROS_INFO_STREAM("getImageService Called");
    sensor_msgs::Image msg = PrepareImageMsg(blackened_cloud);
    res.msg = msg;
    return true;
}
bool ImagePreparator::UpdateTFs(deep_cnn_object_detection::UpdateTFs::Request& req, deep_cnn_object_detection::UpdateTFs::Response& res)
{
    ROS_INFO_STREAM("UpdateTFs service Called");
    transform_world_camera = GetTransform("/world", "/camera_rgb_optical_frame");
//     transform_world_rotary_table = GetTransform("/world", "/rotary_table");
    return true;
}
void ImagePreparator::PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& pc_msg)
{
    ROS_INFO_STREAM("Cloud received");
    input_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*pc_msg, *this->input_cloud);
    MakeImage();
//     ros::Duration(1.0).sleep();
}
void ImagePreparator::MakeImage()
{   
//     transform_world_camera = GetTransform("/world", "/camera_rgb_optical_frame");
    transform_world_rotary_table = GetTransform("/world", "/rotary_table");
    ok = true;
    blackened_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (input_cloud->points.size() > 0)
    {
	TransformPointCloud(input_cloud, transformed_cloud, TransformEigen_world_camera);
	*blackened_cloud = *transformed_cloud;
	CutCubeAroundTable();   
// 	if (!FindAndRemoveMainPlane())
// 	{
// 	    ROS_ERROR_STREAM("Could not find and remove plane");
// 	}
// 	else
// 	{
// 	    if (!FindMainCluster())
// 	    {
// 		ROS_ERROR_STREAM("Could not find object cluster");
// 	    }
// 	    else
// 	    {
// 		BlackCube();
// 		PrepareImageMsg(blackened_cloud);
// 	    }
// 	}	
	    BlackCube();
	    PrepareImageMsg(blackened_cloud);
    }
    else 
    {
	ROS_ERROR_STREAM("Input cloud size = 0");
    }
}
void ImagePreparator::CutCubeAroundTable()
{
    float x = transform_world_rotary_table.getOrigin().x();
    float y = transform_world_rotary_table.getOrigin().y();
    float z = transform_world_rotary_table.getOrigin().z();
    
    float dx = 0.3;
    float dy = 0.3;
    float nz = 0.02, pz = 0.4;
    
    PassThroughFilter(transformed_cloud, cutted_cloud, "x", x - dx, x + dx);
    PassThroughFilter(cutted_cloud, cutted_cloud, "y", y - dy, y + dy);
    PassThroughFilter(cutted_cloud, cutted_cloud, "z", z - nz, z + pz);
}
bool ImagePreparator::FindAndRemoveMainPlane()
{
    pcl::ModelCoefficients::Ptr PlaneCoefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr PlaneInliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> PlaneFinder;
    PlaneFinder.setOptimizeCoefficients (true);
    PlaneFinder.setModelType (pcl::SACMODEL_PLANE);
    PlaneFinder.setMethodType (pcl::SAC_RANSAC);
    PlaneFinder.setDistanceThreshold (0.008);
    PlaneFinder.setInputCloud (cutted_cloud);
    PlaneFinder.segment (*PlaneInliers, *PlaneCoefficients);
    
    if (PlaneInliers->indices.size() > 0)
    {    
	pcl::ExtractIndices<pcl::PointXYZRGB> Extractor;
	Extractor.setInputCloud (cutted_cloud);
	Extractor.setIndices (PlaneInliers);
	Extractor.setNegative (true);
	Extractor.filter (*object_cloud);
    }
    else 
    {
	return false;
    }
    return true;
}
bool ImagePreparator::FindMainCluster()
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (object_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ClusterExtractor;
    ClusterExtractor.setClusterTolerance (0.01); // 2cm
    ClusterExtractor.setMinClusterSize (500);
    ClusterExtractor.setMaxClusterSize (100000);
    ClusterExtractor.setSearchMethod (tree);
    ClusterExtractor.setInputCloud (object_cloud);
    ClusterExtractor.extract (cluster_indices);
    
    if (cluster_indices.size() > 0)
    {   
	pcl::PointIndices::Ptr ClusterInliers (new pcl::PointIndices(cluster_indices.at(0)));    
	pcl::ExtractIndices<pcl::PointXYZRGB> Extractor;
	Extractor.setInputCloud (object_cloud);
	Extractor.setIndices (ClusterInliers);
	Extractor.setNegative (false);
	Extractor.filter (*object_cloud_filtered);
    }
    else 
    {
	return false;
    }
    return true;
}
void ImagePreparator::BlackCube()
{
    MakeCloudBlackened(transformed_cloud, cutted_cloud);
}
tf::StampedTransform ImagePreparator::GetTransform(std::string parent, std::string child)
{
    tf::TransformListener listener;
    tf::StampedTransform out;
    try
    {
// 	ROS_INFO_STREAM("Waiting for transform");
	if (listener.waitForTransform(parent, child, ros::Time(0), ros::Duration(5)))
	{
// 	    ROS_INFO_STREAM("Getting transform");
	    listener.lookupTransform(parent, child, ros::Time(0), out); 
// 	    ROS_INFO_STREAM("Transform success");
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
void ImagePreparator::TransformPointCloud(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointCloud< pcl::PointXYZRGB >::Ptr output, Eigen::Affine3d transform)
{
    pcl::transformPointCloud(*input, *output, transform);
}
void ImagePreparator::VoxelGridFiter(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointCloud< pcl::PointXYZRGB >::Ptr output, float LEAF_SIZE)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> VoxelGridFIlter;
    VoxelGridFIlter.setInputCloud (input);
    VoxelGridFIlter.setLeafSize (LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    VoxelGridFIlter.filter (*output);
}
void ImagePreparator::PassThroughFilter(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointCloud< pcl::PointXYZRGB >::Ptr output, std::string field, float nx, float px)
{
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (input);
    pass.setFilterFieldName (field);
    pass.setFilterLimits (nx, px);
    pass.filter (*output);    
}
void ImagePreparator::PassThroughFilter(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointCloud< pcl::PointXYZRGB >::Ptr output, std::string field, float nx, float px, pcl::PointIndices::Ptr inliers)
{
    pcl::PassThrough<pcl::PointXYZRGB> pass(true);
    pass.setInputCloud (input);
    pass.setFilterFieldName (field);
    pass.setFilterLimits (nx, px);
    pass.filter (*output);    
    pass.getRemovedIndices(*inliers);
}
void ImagePreparator::MakeCloudBlackened(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointIndices::Ptr inliers)
{
    for (size_t i = 0; i < inliers->indices.size(); i++)
    {
	input->points.at(inliers->indices.at(i)).r = 124;
	input->points.at(inliers->indices.at(i)).g = 117;
	input->points.at(inliers->indices.at(i)).b = 104;
    }
}
void ImagePreparator::MakeCloudBlackened(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::ModelCoefficients::Ptr PlaneCoefficients)
{
    float A = PlaneCoefficients->values[0],
          B = PlaneCoefficients->values[1],
          C = PlaneCoefficients->values[2],
          D = PlaneCoefficients->values[3];	  
    for (size_t i = 0; i < input->points.size(); i++)
    {
	if (input->points.at(i).r != 0)
	{
	    float x = input->points.at(i).x,
		y = input->points.at(i).y,
		z = input->points.at(i).z;
	    float res = A * x + B * y + C * z + D;
	    if(fabs(res) < 0.008)
	    {
		input->points.at(i).r = 124;
		input->points.at(i).g = 117;
		input->points.at(i).b = 104;
	    }
	}
    }
}
void ImagePreparator::MakeCloudBlackened(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input, pcl::PointCloud< pcl::PointXYZRGB >::Ptr object)
{
    pcl::PointXYZRGB min_point, max_point;
    pcl::getMinMax3D(*object, min_point, max_point);
    
    pcl::PointIndices::Ptr indices (new pcl::PointIndices); 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trash (new pcl::PointCloud<pcl::PointXYZRGB>);
    PassThroughFilter(blackened_cloud, trash, "x", min_point.x, max_point.x, indices);
    MakeCloudBlackened(blackened_cloud, indices);  
    PassThroughFilter(blackened_cloud, trash, "y", min_point.y, max_point.y, indices);
    MakeCloudBlackened(blackened_cloud, indices);  
    PassThroughFilter(blackened_cloud, trash, "z", min_point.z, max_point.z, indices);
    MakeCloudBlackened(blackened_cloud, indices);  
}
sensor_msgs::Image ImagePreparator::PrepareImageMsg(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input)
{
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)); // BGR!!!
    for (size_t i = 0; i < input->points.size(); i++)
    {		
	int h = i / 640;
	int w = i % 640;
	image.at<cv::Vec3b>(h, w)[0] = (int)input->points.at(i).b;
	image.at<cv::Vec3b>(h, w)[1] = (int)input->points.at(i).g;
	image.at<cv::Vec3b>(h, w)[2] = (int)input->points.at(i).r;  
    }
    cv_bridge::CvImage out_msg;
    out_msg.header.frame_id = "camera_rgb_optical_frame";
    out_msg.header.stamp = ros::Time::now();
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = image;
    image_publisher.publish(out_msg.toImageMsg());
    return *out_msg.toImageMsg();
}
void ImagePreparator::VisualizeCloud(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input)
{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(input);
    pcl::visualization::PCLVisualizer visualizer("Visualiser");
    visualizer.addCoordinateSystem(0.2);
    visualizer.setBackgroundColor(0.0,0.0,0.0);
    visualizer.addPointCloud(input, rgb, "CloudInput");

    while(!visualizer.wasStopped())
    {
	visualizer.spinOnce();
    }
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_two_ints_server");
    ros::NodeHandle n;
    ImagePreparator image_preparator;
    ros::ServiceServer getImage_service = n.advertiseService("image_preparator/getImage", &ImagePreparator::getImage, &image_preparator);      
    ros::ServiceServer UpdateTFs_service = n.advertiseService("image_preparator/UpdateTFs", &ImagePreparator::UpdateTFs, &image_preparator);      
    
    while (ros::ok())
    {
	ros::spinOnce();
	ros::Duration(0.2).sleep();
    }

    return 0;
}
