#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/registration/icp.h>

#include <pcl/visualization/pcl_visualizer.h>

//#include <pcl/features/fpfh.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/sample_consensus/model_types.h>
//#include <pcl/registration/ia_ransac.h>
//#include <pcl/registration/icp.h>

typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> ColorHandlerTXYZRGB;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerTXYZ;

struct OURCVFHModel
{
    std::string Name;
    std::vector<float> Histogram;
};
class PointCloudCluster
{
public:
    PointCloudCluster();
    void SetInput(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _CloudInput);
    std::vector<OURCVFHModel> GetOURCVFHModels();
    std::vector<bool> GetValidTransformationsMask();
    std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > GetTransformations();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetCloudSmoothed();
private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudInput;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudVoxeled;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudSmoothed;
    pcl::PointCloud<pcl::Normal>::Ptr CloudNormals;

    std::vector<OURCVFHModel> OURCVFHModels;

    std::vector<bool> ValidTransformationsMask;
    std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > Transformations;

    void ProcessCLoud();
    void VoxelFilter();
    void MLSFIlter();
    void RemoveNANs();
    void ComputeNormals();
    void ComputeOURCVFHModels();
};

class PointCloudProcessor
{
    public:
        PointCloudProcessor();
        void SetInputCloud(boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > __input_cloud);
        std::vector<PointCloudCluster> GetClusters();
    private:
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudInput;
        std::vector<PointCloudCluster> Clusters;

        void ProcessInputCloud();
        void FindPlane();
        void DeletePlane(pcl::ModelCoefficients::Ptr _PlaneCoefficients, pcl::PointIndices::Ptr _PlaneInliers);
        void FindClusters();
};

