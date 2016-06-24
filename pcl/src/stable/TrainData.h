#include <vector>
#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include "PCLProcessor.h"
#include "../experimental/db_msd_pcl.h"

#include <pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/hv/hv_papazov.h>



#include <ros/ros.h>
#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/common/types.h>
#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>

typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerTXYZ;

class Object
{
public:
    void SetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr _Cloud);
    void SetID(std::string _ID);
    std::string GetID();
    void SetTrainingData(object_recognition_core::db::data _TrainingData);
    object_recognition_core::db::data GetTrainingData();
    pcl::PointCloud<pcl::PointXYZ>::Ptr GetCloud();
private:
    std::string ID;
    pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud;
    object_recognition_core::db::data TrainingData;
};
struct IndexScore
{
    int idx_models_;
    int idx_input_;
    double score_;
};
struct sortIndexScores
{
    bool operator() (const IndexScore& d1, const IndexScore& d2)
    {
        return d1.score_ < d2.score_;
    }
} sortIndexScoresOp;
class Detector
{
public:
    Detector();
    void AddObject(Object _Object);
    void BuildFlannIndex();
    void SetDB(object_recognition_core::db::ObjectDbPtr & db);
    void setClusters(std::vector<PointCloudCluster> _Clusters);
    void DetectObject();
    int GetObjectsNumber();
    std::vector<object_recognition_core::common::PoseResult> GetPoseReuslts();

private:
    void PrepareOURCVFHData();
    void FindKNeighbours(PointCloudCluster Cluster);
    void EstimatePositions(PointCloudCluster Cluster);
    void PoseRefinement(PointCloudCluster Cluster);
    void HypothesesVerification();
    void MSDHypothesesVerification();
    void ComputePoseResults(boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > TransformationsFinal, int Winner);
    inline void NearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const OURCVFHModel &model, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

    int NeighboursNumber;
    object_recognition_core::db::ObjectDbPtr db_;

    std::vector<Object> Objects;
    std::vector<PointCloudCluster> Clusters;
    std::vector<object_recognition_core::common::PoseResult> PoseResults;

    std::vector<IndexScore> IndexScores;
    flann::Index<flann::ChiSquareDistance<float> > *OURCVFHIndex;
    object_recognition_core::db::data CommonTrainingData;
    std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> ObjectsClouds;
    pcl::PointCloud<pcl::PointXYZ>::Ptr CloudSensorInput;
    pcl::PointCloud<pcl::PointXYZ>::Ptr CloudSensorInput_HV;
    std::vector<std::string> PosesList;
    std::vector<std::string> HistogramsList;
    std::vector<int> ModelList;

    std::vector<OURCVFHModel> OURCVFHModels;
    boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > Transformations;
};
