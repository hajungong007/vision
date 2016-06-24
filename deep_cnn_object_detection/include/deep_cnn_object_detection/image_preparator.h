#ifndef IMAGE_PREPARATOR_H_
#define IMAGE_PREPARATOR_H_
#include <deep_cnn_object_detection/getImage.h>
#include <deep_cnn_object_detection/UpdateTFs.h>

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

#include <tf/transform_listener.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf_conversions/tf_eigen.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <moveit/background_processing/background_processing.h>

class ImagePreparator
{
public:
    ImagePreparator();
    bool getImage(deep_cnn_object_detection::getImage::Request  &req,
		  deep_cnn_object_detection::getImage::Response &res);
    bool UpdateTFs(deep_cnn_object_detection::UpdateTFs::Request &req,
		   deep_cnn_object_detection::UpdateTFs::Response &res);
private:
    bool ok;
    ros::NodeHandle nh_;
    std::string subs_cloud_topic_;
    ros::Subscriber pc_subs_;  
    ros::Publisher image_publisher;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cutted_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud_filtered;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr blackened_cloud;
    
    tf::StampedTransform transform_world_camera;
    tf::StampedTransform transform_world_rotary_table;
    Eigen::Affine3d TransformEigen_world_camera;
    Eigen::Affine3d TransformEigen_camera_world;
    
    void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &pc_msg);
    void MakeImage();
    void BlackCube();
    void CutCubeAroundTable();
    bool FindAndRemoveMainPlane();
    bool FindMainCluster();
    
    void VoxelGridFiter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output, float LEAF_SIZE);
    void TransformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output, Eigen::Affine3d transform);
    void PassThroughFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output, std::string field, float nx, float px);
    void PassThroughFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output, std::string field, float nx, float px, pcl::PointIndices::Ptr inliers);
    void MakeCloudBlackened(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointIndices::Ptr inliers);
    void MakeCloudBlackened(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::ModelCoefficients::Ptr PlaneCoefficients);
    void MakeCloudBlackened(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr object);
    sensor_msgs::Image PrepareImageMsg(pcl::PointCloud< pcl::PointXYZRGB >::Ptr input);
    void VisualizeCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input);
    
    tf::StampedTransform GetTransform(std::string parent, std::string child);
};

#endif