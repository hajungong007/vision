#include "PCLProcessor.h"

PointCloudCluster::PointCloudCluster()
{
    CloudInput = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    CloudVoxeled = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    CloudSmoothed = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    CloudNormals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);
}
void PointCloudCluster::SetInput(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _CloudInput)
{
   *CloudInput = *_CloudInput;
    ProcessCLoud();
}
std::vector<OURCVFHModel> PointCloudCluster::GetOURCVFHModels()
{
    std::vector<OURCVFHModel> _OURCVFHModels;
    _OURCVFHModels = OURCVFHModels;
    return _OURCVFHModels;
}
std::vector<bool> PointCloudCluster::GetValidTransformationsMask()
{
    std::vector<bool> _mask;
    _mask = ValidTransformationsMask;
    return _mask;
}
std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > PointCloudCluster::GetTransformations()
{
    std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > _Transformations;
    _Transformations = Transformations;
    return _Transformations;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudCluster::GetCloudSmoothed()
{
    return CloudSmoothed;
}
void PointCloudCluster::ProcessCLoud()
{
    VoxelFilter();
    MLSFIlter();
    RemoveNANs();
    ComputeNormals();
    ComputeOURCVFHModels();
}
void PointCloudCluster::VoxelFilter()
{
    float VOXEL_LEAF_SIZE = 0.005;
    pcl::VoxelGrid<pcl::PointXYZRGB> VoxelGridFIlter;
    VoxelGridFIlter.setInputCloud (CloudInput);
    VoxelGridFIlter.setLeafSize (VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE);
    VoxelGridFIlter.filter (*CloudVoxeled);
}
void PointCloudCluster::MLSFIlter()
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr Tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> MLSFIlter;
    MLSFIlter.setComputeNormals (false);
    MLSFIlter.setPolynomialFit (true);
    MLSFIlter.setSearchMethod (Tree);
    MLSFIlter.setSearchRadius (0.015);
    MLSFIlter.setInputCloud (CloudVoxeled);
    MLSFIlter.process (*CloudSmoothed);

}
void PointCloudCluster::RemoveNANs()
{
    for (size_t i = 0; i < CloudSmoothed->size(); i++)
    {
        if (isnan(CloudSmoothed->points[i].x))
        {
            CloudSmoothed->points[i].x = CloudSmoothed->points[i - 1].x;
            CloudSmoothed->points[i].y = CloudSmoothed->points[i - 1].y;
            CloudSmoothed->points[i].z = CloudSmoothed->points[i - 1].z;
        }
    }
}
void PointCloudCluster::ComputeNormals()
{
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> NormalEstimator;
    NormalEstimator.setRadiusSearch(0.020);
    NormalEstimator.setInputCloud (CloudSmoothed);
    NormalEstimator.compute (*CloudNormals);
}
void PointCloudCluster::ComputeOURCVFHModels()
{
    pcl::OURCVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> OURCVFHEstimator;

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr Tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    OURCVFHEstimator.setSearchMethod (Tree);
    // Compute the features
    OURCVFHEstimator.setEPSAngleThreshold(0.13f); // 5 degrees.
    OURCVFHEstimator.setCurvatureThreshold(0.026f);
    OURCVFHEstimator.setClusterTolerance (0.015f);
    OURCVFHEstimator.setNormalizeBins(false);
    OURCVFHEstimator.setAxisRatio(0.8);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr OURCVFHSignature (new pcl::PointCloud<pcl::VFHSignature308> ());
    OURCVFHEstimator.setInputCloud (CloudSmoothed);
    OURCVFHEstimator.setInputNormals (CloudNormals);
    std::cout << "Computing OURCVFH\n";
    OURCVFHEstimator.compute (*OURCVFHSignature);
    std::cout << "OURCVFH computed\n";
    OURCVFHEstimator.getTransforms(Transformations);
    OURCVFHEstimator.getValidTransformsVec(ValidTransformationsMask);



    for (uint hist_id = 0; hist_id < OURCVFHSignature->points.size(); hist_id++)
    {
        OURCVFHModel _OURCVFHModel;
        std::stringstream cvfh_ss;
        cvfh_ss << hist_id;
        _OURCVFHModel.Name = cvfh_ss.str();
        _OURCVFHModel.Histogram.resize(308);

        for (size_t i = 0; i < 308; ++i)
        {
            _OURCVFHModel.Histogram[i] = OURCVFHSignature->points[hist_id].histogram[i];
        }
        OURCVFHModels.push_back(_OURCVFHModel);
    }
}

PointCloudProcessor::PointCloudProcessor()
{
    CloudInput = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
}
void PointCloudProcessor::SetInputCloud(boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> > __input_cloud)
{
    *CloudInput = *__input_cloud;
    ProcessInputCloud();
}
void PointCloudProcessor::ProcessInputCloud()
{
    FindPlane();
    FindClusters();
}
void PointCloudProcessor::FindPlane()
{
    pcl::ModelCoefficients::Ptr PlaneCoefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr PlaneInliers (new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZRGB> PlaneFinder;

    PlaneFinder.setOptimizeCoefficients (true);

    PlaneFinder.setModelType (pcl::SACMODEL_PLANE);
    PlaneFinder.setMethodType (pcl::SAC_RANSAC);
    PlaneFinder.setDistanceThreshold (0.01);

    PlaneFinder.setInputCloud (CloudInput);
    PlaneFinder.segment (*PlaneInliers, *PlaneCoefficients);

}
void PointCloudProcessor::DeletePlane(pcl::ModelCoefficients::Ptr _PlaneCoefficients, pcl::PointIndices::Ptr _PlaneInliers)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudTemp (new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::ExtractIndices<pcl::PointXYZRGB> Extractor;
    Extractor.setInputCloud (CloudInput);
    Extractor.setIndices (_PlaneInliers);
    Extractor.setNegative (true);
    Extractor.filter (*CloudTemp);

    CloudInput->clear();

    float A = _PlaneCoefficients->values[0],
          B = _PlaneCoefficients->values[1],
          C = _PlaneCoefficients->values[2],
          D = _PlaneCoefficients->values[3];
    for (size_t i = 0; i < CloudTemp->points.size(); i++)
    {
        float Plane = A * CloudTemp->points[i].x +
                      B * CloudTemp->points[i].y +
                      C * CloudTemp->points[i].z + D;
        if (C > 0)
        {
            if (Plane < 0)
            {
                CloudInput->points.push_back(CloudTemp->points[i]);
            }
        }
        else
        {
            if (Plane > 0)
            {
                CloudInput->points.push_back(CloudTemp->points[i]);
            }
        }
    }
}
void PointCloudProcessor::FindClusters()
{
    Clusters.clear();
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr Tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    Tree->setInputCloud (CloudInput);
    std::vector<pcl::PointIndices> ClustersIndicies;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ClusterExtractor;
    ClusterExtractor.setClusterTolerance (0.02); // 2cm
    ClusterExtractor.setMinClusterSize (100);
    ClusterExtractor.setMaxClusterSize (25000);
    ClusterExtractor.setSearchMethod (Tree);
    ClusterExtractor.setInputCloud (CloudInput);
    ClusterExtractor.extract (ClustersIndicies);

    for (std::vector<pcl::PointIndices>::const_iterator it = ClustersIndicies.begin (); it != ClustersIndicies.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudTemp (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            CloudTemp->points.push_back (CloudInput->points[*pit]); //*
        CloudTemp->width = CloudTemp->points.size ();
        CloudTemp->height = 1;
        CloudTemp->is_dense = true;

        PointCloudCluster Cluster;
        Cluster.SetInput(CloudTemp);
        Clusters.push_back(Cluster);
    }

    std::cout << "Clusters size: " << Clusters.size() << "\n";
}
std::vector<PointCloudCluster> PointCloudProcessor::GetClusters()
{
    std::vector<PointCloudCluster> _Clusters = Clusters;
    return _Clusters;
}












//    pcl::visualization::PCLVisualizer visualizer("Visualiser");
//    visualizer.addCoordinateSystem(0.2);
//    visualizer.setBackgroundColor(0.0,0.0,0.0);
//    visualizer.addPointCloud(CloudInput,
//                             ColorHandlerTXYZRGB(CloudInput, 255.0, 0, 0.0),
//                             "CloudInput");

//    while(!visualizer.wasStopped())
//    {
//        visualizer.spinOnce();
//    }
