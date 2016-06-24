#include "TrainData.h"

void Object::SetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr _Cloud)
{
    Cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    *Cloud = *_Cloud;
}
void Object::SetID(std::string _ID)
{
    ID = _ID;
}
void Object::SetTrainingData(object_recognition_core::db::data _TrainingData)
{
    TrainingData = _TrainingData;
}
object_recognition_core::db::data Object::GetTrainingData()
{
    object_recognition_core::db::data _TrainingData;
    _TrainingData = TrainingData;
    return _TrainingData;
}
std::string Object::GetID()
{
    std::string _ID;
    _ID = ID;
    return _ID;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr Object::GetCloud()
{
    return Cloud;
}
Detector::Detector()
{
    NeighboursNumber = 8;
    CloudSensorInput = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    CloudSensorInput_HV = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
}
void Detector::AddObject(Object _Object)
{
    Objects.push_back(_Object);
}
void Detector::SetDB(object_recognition_core::db::ObjectDbPtr &db)
{
    db_ = db;
}
void Detector::setClusters(std::vector<PointCloudCluster> _Clusters)
{
    Clusters = _Clusters;
}
void Detector::PrepareOURCVFHData()
{
    ObjectsClouds.resize(Objects.size());
    for (size_t i = 0; i < Objects.size(); i++)
    {
        object_recognition_core::db::data _TrainingData = Objects[i].GetTrainingData();
        for (size_t j = 0; j < _TrainingData.histograms.size(); ++j)
        {
            HistogramsList.push_back(Objects[i].GetID());
            CommonTrainingData.histograms.push_back(_TrainingData.histograms[j]);
            CommonTrainingData.centroids.push_back(_TrainingData.centroids[j]);
            CommonTrainingData.roll_transforms.push_back(_TrainingData.roll_transforms[j]);
            CommonTrainingData.ids.push_back(_TrainingData.ids[j]);
            ModelList.push_back(i);
        }
        for (size_t j = 0; j < _TrainingData.poses.size(); ++j)
        {
            PosesList.push_back(Objects[i].GetID());
            CommonTrainingData.poses.push_back(_TrainingData.poses[j]);
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr CloudTemp (new pcl::PointCloud<pcl::PointXYZ>);
        CloudTemp = Objects[i].GetCloud();
        ObjectsClouds[i] = CloudTemp;
    }
}
void Detector::BuildFlannIndex()
{
    PrepareOURCVFHData();
    flann::Matrix<float> OURCVFHData (new float[CommonTrainingData.histograms.size () * CommonTrainingData.histograms[0].size ()],
                                        CommonTrainingData.histograms.size (),
                                        CommonTrainingData.histograms[0].size ());
    for (size_t i = 0; i < CommonTrainingData.histograms.size(); ++i)
    {
        OURCVFHModel _OURCVFHModel;
        _OURCVFHModel.Name = CommonTrainingData.ids[i];
        for (size_t j = 0; j < CommonTrainingData.histograms[i].size(); ++j)
        {
            OURCVFHData[i][j] = CommonTrainingData.histograms[i][j];
        }
        _OURCVFHModel.Histogram = CommonTrainingData.histograms[i];
        OURCVFHModels.push_back(_OURCVFHModel);

    }
    OURCVFHIndex = new  flann::Index<flann::ChiSquareDistance<float> >(OURCVFHData, flann::LinearIndexParams ());
    OURCVFHIndex->buildIndex();
}
void Detector::DetectObject()
{
    PoseResults.clear();
    BOOST_FOREACH(PointCloudCluster & Cluster, Clusters)
    {

        std::cout << "FindKNeighbours \n";
        FindKNeighbours(Cluster);
        std::cout << "EstimatePositions \n";
        EstimatePositions(Cluster);
        std::cout << "PoseRefinement \n";
        PoseRefinement(Cluster);
        std::cout << "HypothesesVerification \n";
        HypothesesVerification();
        std::cout << "Done \n";
    }
}
void Detector::FindKNeighbours(PointCloudCluster Cluster)
{
    IndexScores.clear();

    std::vector<OURCVFHModel> _OURCVFHModels;
    _OURCVFHModels = Cluster.GetOURCVFHModels();

    flann::Matrix<int> Indices;
    flann::Matrix<float> Distances;
    
    std::cout << "_OURCVFHModels.size(): " <<_OURCVFHModels.size() << " \n";
    
    for (uint i = 0; i < _OURCVFHModels.size(); i++)
    {
        OURCVFHModel _OURCVFHModel;
        _OURCVFHModel = _OURCVFHModels[i];
        NearestKSearch (*OURCVFHIndex, _OURCVFHModel, NeighboursNumber, Indices, Distances);
        for (int NN_ = 0; NN_ < NeighboursNumber; ++NN_)
        {
            IndexScore is;
            is.idx_models_ = Indices[0][NN_];
            is.idx_input_ = static_cast<int> (i);
            is.score_ = Distances[0][NN_];
            IndexScores.push_back (is);
        }
    }
}
inline void Detector::NearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const OURCVFHModel &model, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
    // Query point
    flann::Matrix<float> p = flann::Matrix<float>(new float[model.Histogram.size ()], 1, model.Histogram.size ());
    memcpy (&p.ptr ()[0], &model.Histogram[0], p.cols * p.rows * sizeof (float));
    indices = flann::Matrix<int>(new int[k], 1, k);
    distances = flann::Matrix<float>(new float[k], 1, k);
    index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
    delete[] p.ptr ();
}
void Detector::EstimatePositions(PointCloudCluster Cluster)
{
    Transformations.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);
    for (uint hyp_num = 0; hyp_num < IndexScores.size(); hyp_num++)
    {
        std::string id_string = OURCVFHModels.at (IndexScores[hyp_num].idx_models_).Name;
        int view_id = atoi(id_string.substr(0, id_string.find_first_of("_")).c_str());
        int idx_input = IndexScores[hyp_num].idx_input_;

        Eigen::Matrix4f RollViewPose;
        RollViewPose = CommonTrainingData.roll_transforms[IndexScores[hyp_num].idx_models_];

        std::vector<bool> ValidTransformsMask;
        ValidTransformsMask = Cluster.GetValidTransformationsMask();
        std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > _Tranformations;
        _Tranformations = Cluster.GetTransformations();

        if(ValidTransformsMask[idx_input])
        {

            Eigen::Matrix4f ModelViewPose;
            ModelViewPose = CommonTrainingData.poses[view_id];

            Eigen::Matrix4f ScaleMatrix;
            ScaleMatrix.setIdentity (4, 4);

            Eigen::Matrix4f Transformation;
            Transformation = ScaleMatrix * _Tranformations[idx_input].inverse () * RollViewPose * ModelViewPose;

            Transformations->push_back (Transformation);
        }
    }
}
void Detector::PoseRefinement(PointCloudCluster Cluster)
{
    CloudSensorInput->clear();
    pcl::PointCloud<pcl::PointXYZ>::Ptr CloudSensorTempXYZ (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudSensorTempXYZRGB (new pcl::PointCloud<pcl::PointXYZRGB>);

    *CloudSensorTempXYZRGB = *Cluster.GetCloudSmoothed();

    CloudSensorTempXYZ->points.resize(CloudSensorTempXYZRGB->size());
    for (size_t i = 0; i < CloudSensorTempXYZRGB->points.size(); i++)
    {
        CloudSensorTempXYZ->points[i].x = CloudSensorTempXYZRGB->points[i].x;
        CloudSensorTempXYZ->points[i].y = CloudSensorTempXYZRGB->points[i].y;
        CloudSensorTempXYZ->points[i].z = CloudSensorTempXYZRGB->points[i].z;
    }

    
    pcl::VoxelGrid<pcl::PointXYZ> VoxelGridFilter;
    VoxelGridFilter.setInputCloud (CloudSensorTempXYZ);
    float VOXEL_SIZE_ICP_ = 0.005;
    VoxelGridFilter.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
    VoxelGridFilter.filter (*CloudSensorInput_HV);
    
    VOXEL_SIZE_ICP_ = 0.008;
    VoxelGridFilter.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
    VoxelGridFilter.filter (*CloudSensorInput);

    std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> CloudsPreAlighned;
    CloudsPreAlighned.resize (Transformations->size());
    for (uint Hypothese = 0; Hypothese < Transformations->size(); Hypothese++)
    {
	std::cout << "Starting alighning of hypothese num: " << Hypothese + 1 << "  \n";
	
        pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModel (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr CLoudModelTemp (new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud (*ObjectsClouds[ModelList[IndexScores[Hypothese].idx_models_]], *CLoudModelTemp, Transformations->at (Hypothese));

        CloudsPreAlighned[Hypothese] = CLoudModelTemp;

        pcl::VoxelGrid<pcl::PointXYZ> VoxelGridFilterICP;
        VoxelGridFilterICP.setInputCloud (CLoudModelTemp);
        VoxelGridFilterICP.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
        VoxelGridFilterICP.filter (*CloudModel);

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> ICPAlighner;
        ICPAlighner.setInputSource(CloudModel); //model
        ICPAlighner.setInputTarget (CloudSensorInput); //scene
        ICPAlighner.setMaximumIterations (200);
        ICPAlighner.setMaxCorrespondenceDistance (0.01);
        ICPAlighner.setTransformationEpsilon (1e-6);

        pcl::PointCloud<pcl::PointXYZ>::Ptr CloudAlighned (new pcl::PointCloud<pcl::PointXYZ> ());
        ICPAlighner.align (*CloudAlighned);

        Eigen::Matrix4f TransformationICP = ICPAlighner.getFinalTransformation ();
        Transformations->at (Hypothese) = TransformationICP * Transformations->at (Hypothese);
    }
    
//     pcl::visualization::PCLVisualizer visualizer("Visualiser");
//     int k = CloudsPreAlighned.size();
//     int y_s = (int)floor (sqrt ((double)k));
//     int x_s = y_s + (int)ceil ((CloudsPreAlighned.size() / (double)y_s) - y_s);
//     double x_step = (double)(1 / (double)x_s);
//     double y_step = (double)(1 / (double)y_s);
// 
// 
//     int viewport = 0, l = 0, m = 0;
//     
//     for (int i = 0; i < k; ++i)
//     {
// 	visualizer.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
// 	l++;
// 	if (l >= x_s)
// 	{
// 	    l = 0;
// 	    m++;
// 	}
// 	std::string cloud_name = "CloudSensorInput_HV" + boost::to_string(i);
// 	visualizer.addPointCloud(CloudSensorInput_HV,
//                             ColorHandlerTXYZ(CloudSensorInput_HV, 0.0, 255.0, 0.0),
//                             cloud_name, viewport);
// 	cloud_name = "CloudsAlighned.at" + boost::to_string(i);
// 	visualizer.addPointCloud(CloudsPreAlighned.at(i),
//                             ColorHandlerTXYZ(CloudsPreAlighned.at(i), 255.0, 0.0, 0.0),
//                             cloud_name, viewport);
// 	
// 	std::stringstream ss;
// 	ss << "Hypothese " << i;
// 	visualizer.addText (ss.str (), 20, 30, 1, 0, 0, ss.str (), viewport);  // display the text with red
//     }
//     visualizer.addCoordinateSystem(0.2);
//     visualizer.setBackgroundColor(0.0,0.0,0.0);
// 
//     while(!visualizer.wasStopped())
//     {
// 	visualizer.spinOnce();
//     }

}
void Detector::HypothesesVerification()
{
    std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> CloudsAlighned;
    std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> CloudsAlighnedTMP;
    CloudsAlighned.resize (Transformations->size());
    if (CloudsAlighned.size() == 0)
        return;
    float VOXEL_SIZE = 0.005;
    
    pcl::VoxelGrid<pcl::PointXYZ> VoxelGridFilter;
    VoxelGridFilter.setLeafSize (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
    
    for (size_t Hypothese = 0; Hypothese < Transformations->size(); Hypothese++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModelAlighned (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModelAlighnedVoxeled (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*ObjectsClouds[ModelList[IndexScores[Hypothese].idx_models_]], *CloudModelAlighned, Transformations->at (Hypothese));
	
	VoxelGridFilter.setInputCloud (CloudModelAlighned);
	VoxelGridFilter.filter (*CloudModelAlighnedVoxeled);
	
        CloudsAlighned[Hypothese] = CloudModelAlighnedVoxeled;
    }
    
    boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > TransformationsFinal;
    TransformationsFinal.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);
//     int Winners = 0;
    int Winner = 0;

    int curr_cloud = 1;
    std::vector<int> outliers_vector;
    std::vector<float> distances;
    float outliers_treshold = 0.006;
    for (std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr>::iterator hypothese_it = CloudsAlighned.begin(); hypothese_it != CloudsAlighned.end(); ++hypothese_it)
    {
	pcl::PointXYZ searchpoint;
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
// 	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	std::cout << "----------------------------------------------------\n";
	std::cout << "Cloud # " << curr_cloud << "\n";
	std::cout << "Model points number: " << (*hypothese_it)->points.size() << "\n";
	std::cout << "Scene points number: " << CloudSensorInput_HV->points.size() << "\n";
	
	int inliers = 0;
	int outliers = 0;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr Tree (new pcl::search::KdTree<pcl::PointXYZ>);
	Tree->setInputCloud (*hypothese_it);
	
	distances.push_back(0);
	for (size_t j = 0; j < CloudSensorInput_HV->points.size(); j++)
	{
	    searchpoint.x = CloudSensorInput_HV->points.at(j).x;
	    searchpoint.y = CloudSensorInput_HV->points.at(j).y;
	    searchpoint.z = CloudSensorInput_HV->points.at(j).z;
	    
	    if (Tree->radiusSearch (searchpoint, 0.2, pointIdxRadiusSearch, pointRadiusSquaredDistance, 1) > 0)
	    {
// 		std::cout << "For point # " << j << " nearest neighbour is in " << pointRadiusSquaredDistance.back() << " meters" << "\n";
		if (sqrt(pointRadiusSquaredDistance.back()) < outliers_treshold)
		{
		    inliers++;
		}
		else
		{
		    outliers++; 
		}
		distances.back() += sqrt(pointRadiusSquaredDistance.back());
	    }
	    else 
	    {
		outliers++; 	
	    }
	}
	distances.back() = distances.back() / (float)CloudSensorInput_HV->points.size();
	
	std::cout << "inliers: " << inliers << "\n";
	std::cout << "outliers: " << outliers << "\n";
	std::cout << "average distance: " << distances.back() << "\n";
	outliers_vector.push_back(outliers);
	curr_cloud++;
    }  
    Winner = 0;
    for (size_t i = 1; i < distances.size(); i++)
    {
	if (distances.at(i) < distances.at(Winner))
	{
	    Winner = i;
	}
    }
    
    std::cout << "Winner: " << Winner << "\n";

    TransformationsFinal->push_back (Transformations->at (Winner));

//     pcl::visualization::PCLVisualizer visualizer("Visualiser");
//     int k = outliers_vector.size();
//     int y_s = (int)floor (sqrt ((double)k));
//     int x_s = y_s + (int)ceil ((outliers_vector.size() / (double)y_s) - y_s);
//     double x_step = (double)(1 / (double)x_s);
//     double y_step = (double)(1 / (double)y_s);
// 
// 
//     int viewport = 0, l = 0, m = 0;
    
//     for (int i = 0; i < k; ++i)
//     {
// 	visualizer.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
// 	l++;
// 	if (l >= x_s)
// 	{
// 	    l = 0;
// 	    m++;
// 	}
// 	std::string cloud_name = "CloudSensorInput_HV" + boost::to_string(i);
// 	visualizer.addPointCloud(CloudSensorInput_HV,
//                             ColorHandlerTXYZ(CloudSensorInput_HV, 0.0, 255.0, 0.0),
//                             cloud_name, viewport);
// 	cloud_name = "CloudsAlighned.at" + boost::to_string(i);
// 	visualizer.addPointCloud(CloudsAlighned.at(i),
//                             ColorHandlerTXYZ(CloudsAlighned.at(i), 255.0, 0.0, 0.0),
//                             cloud_name, viewport);
// 	
// 	std::stringstream ss;
// 	ss << "Hypothese " << i << ": outliers: " << outliers_vector[i] << "distance: " << distances[i] ;
// 	visualizer.addText (ss.str (), 20, 30, 1, 0, 0, ss.str (), viewport);  // display the text with red
//     }
//     visualizer.addCoordinateSystem(0.2);
//     visualizer.setBackgroundColor(0.0,0.0,0.0);
// 
//     while(!visualizer.wasStopped())
//     {
// 	visualizer.spinOnce();
//     }
//    visualizer.addCoordinateSystem(0.2);
//    visualizer.setBackgroundColor(0.0,0.0,0.0);
//    visualizer.addPointCloud(CloudInput,
//                             ColorHandlerTXYZRGB(CloudInput, 255.0, 0, 0.0),
//                             "CloudInput");
// 
//    while(!visualizer.wasStopped())
//    {
//        visualizer.spinOnce();
//    }
    
    
//     do
//     {
// 	TransformationsFinal->clear();
// 	std::vector<bool> Mask;
// 	pcl::GlobalHypothesesVerification<pcl::PointXYZ, pcl::PointXYZ>  HypothesesVerificator;
// 	HypothesesVerificator.setResolution (0.005f);
// 	HypothesesVerificator.setMaxIterations (7000);
// 	HypothesesVerificator.setInlierThreshold (0.005f);
// 	HypothesesVerificator.setRadiusClutter (0.04f);
// 	HypothesesVerificator.setRegularizer (3.f);
// 	HypothesesVerificator.setClutterRegularizer (7.5f);
// 	HypothesesVerificator.setDetectClutter (0);
// 	HypothesesVerificator.setOcclusionThreshold (0.01f);
// 	HypothesesVerificator.setSceneCloud (CloudSensorInput_HV);
// 	HypothesesVerificator.addModels (CloudsAlighned, true);
// 	HypothesesVerificator.verify ();
// 	HypothesesVerificator.getMask (Mask);
// 
// // 	boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > TransformationsFinal;
// // 	TransformationsFinal.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);
// 
// 	Winner = 0;
// 	Winners = 0;
// 	for (size_t i = 0; i < Mask.size(); i++)
// 	{
// 	    bool mask_ = Mask[i];
// 	    std::cout << "Mask " << i << " : " << mask_ << "\n";
// 	    if (Mask[i])
// 	    {
// 		CloudsAlighnedTMP.push_back(CloudsAlighned.at(i));
// 		TransformationsFinal->push_back (Transformations->at (i));
// 		Winner = i;
// 		Winners++;
//     	    break;
// 	    }
// 	}
// 	CloudsAlighned.clear();
// 	CloudsAlighned = CloudsAlighnedTMP;
// 	std::cout << "Winners: " << Winners  << "\n";
//     }
//     while (Winners > 1);
    
    ComputePoseResults(TransformationsFinal, Winner);
}
void Detector::MSDHypothesesVerification()
{

}
void Detector::ComputePoseResults(boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > TransformationsFinal, int Winner)
{
    BOOST_FOREACH(Eigen::Matrix4f & Transformation, *TransformationsFinal)
    {
        Eigen::Matrix3f FinalMatrix;
        FinalMatrix << Transformation(0,0), Transformation(1,0), Transformation(2,0),
                Transformation(0,1), Transformation(1,1), Transformation(2,1),
                Transformation(0,2), Transformation(1,2), Transformation(2,2);
        Eigen::Vector3f FinalVector(Transformation(0,3), Transformation(1,3), Transformation(2,3));

        object_recognition_core::common::PoseResult PoseResult;
        PoseResult.set_R(FinalMatrix);
        PoseResult.set_T(FinalVector);
        PoseResult.set_object_id(db_, HistogramsList[IndexScores[Winner].idx_models_]);
        PoseResults.push_back(PoseResult);
    }
    if (PoseResults.size() == 1)
        std::cout << "Found " << PoseResults.size() << " object.\n";
    else if (PoseResults.size() > 1)
        std::cout << "Found " << PoseResults.size() << " objects.\n";
}
int Detector::GetObjectsNumber()
{
    return Objects.size();
}
std::vector<object_recognition_core::common::PoseResult> Detector::GetPoseReuslts()
{
    return PoseResults;
}
