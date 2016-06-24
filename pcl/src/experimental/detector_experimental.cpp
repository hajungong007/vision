#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>

#include <ecto/ecto.hpp>
#include <ecto_pcl/ecto_pcl.hpp>
#include <ecto_pcl/pcl_cell.hpp>

#include <Eigen/StdVector>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/esf.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>

//#include <pcl/recognition/hv/hypotheses_verification.h>
#include <pcl/recognition/hv/hv_go.h>

#include <tf/transform_broadcaster.h>

#include <boost/filesystem.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/common/types.h>
#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>
#include <object_recognition_core/db/ModelReader.h>

#include <object_recognition_msgs/RecognizedObjectArray.h>

#include "db_msd_pcl.h"
#include "persistence_utils.h"
#include "vtk_model_sampling.h"

using ecto::tendrils;

using ecto::spore;
//using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectId;
using object_recognition_core::db::DocumentId;

typedef std::pair<std::string, std::vector<float> > vfh_model;
typedef std::pair<std::string, std::vector<float> > cvfh_model;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> ColorHandlerTXYZRGB;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerTXYZ;

namespace ecto_msd_pcl
{
    struct index_score
    {
        int idx_models_;
        int idx_input_;
        double score_;
    };
    struct sortIndexScores
    {
        bool operator() (const index_score& d1, const index_score& d2)
        {
            return d1.score_ < d2.score_;
        }
    } sortIndexScoresOp;
    class PCLCluster
    {
    public:
        std::vector<cvfh_model> cvfh_models;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxeled_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothed_cloud;
        pcl::PointCloud<pcl::Normal>::Ptr normals;

        std::vector<bool> valid_trans;
        std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;

        PCLCluster()
        {
            input_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
            voxeled_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
            smoothed_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
            normals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);
        };
        int a;
    };

    std::vector<Eigen::Vector3f> centroids;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > roll_transforms;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
    std::vector<std::string> ids;
    std::vector<std::vector<float> > histograms;

    struct TrainingData
    {


    };


    struct Detection_experimental : public object_recognition_core::db::bases::ModelReaderBase
    {
        virtual void parameter_callback(const object_recognition_core::db::Documents& db_documents)
        {
            int count = 0;
            model_clouds.resize(db_documents.size());
            BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents)
            {
                std::cout << "------------------- Loading Objects ------------------\n";
                std::string object_id = document.get_field<ObjectId>("object_id");
                object_recognition_core::db::data readed_data;
                document.get_attachment<object_recognition_core::db::data>("data", readed_data);
                for (size_t i = 0; i < readed_data.histograms.size(); ++i)
                {
                    hist_object_ids.push_back(object_id);
                    data_.histograms.push_back(readed_data.histograms[i]);
                    data_.centroids.push_back(readed_data.centroids[i]);
                    data_.roll_transforms.push_back(readed_data.roll_transforms[i]);
                    data_.ids.push_back(readed_data.ids[i]);
                    model_num.push_back(count);
                }
                for (size_t i = 0; i < readed_data.poses.size(); ++i)
                {
                    poses_object_ids.push_back(object_id);
                    data_.poses.push_back(readed_data.poses[i]);
                }
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud__ (new pcl::PointCloud<pcl::PointXYZ>);
                document.get_attachment<pcl::PointCloud<pcl::PointXYZ> >("model_cloud", *cloud__);

//                pcl::visualization::PCLVisualizer visualizer("Visualiser");
//                visualizer.addCoordinateSystem(0.2);
//                visualizer.setBackgroundColor(0.0,0.0,0.0);
//                visualizer.addPointCloud(cloud__,
//                                         ColorHandlerTXYZ(cloud__, 255.0, 0, 0.0),
//                                         "model_aligned_m");

//                while(!visualizer.wasStopped())
//                {
//                    visualizer.spinOnce();
//                }
                model_clouds[count] = cloud__;
                count++;
            }

            std::cout << "hist_object_ids: " << hist_object_ids.size() << "\n";
            std::cout << "poses_object_ids: " << poses_object_ids.size() << "\n";
            std::cout << "Histograms: " << data_.histograms.size() << "\n";
            std::cout << "roll_transform: " << data_.roll_transforms.size() << "\n";
            std::cout << "poses: " << data_.poses.size() << "\n";
            std::cout << "centroids: " << data_.centroids.size() << "\n";
            std::cout << "ids: " << data_.ids.size() << "\n";
            std::cout << "model_clouds: " << model_clouds.size() << "\n";


            if (model_clouds.size() > 0)
            {
//                pcl::visualization::PCLVisualizer visualizer("Visualiser");
//                visualizer.addCoordinateSystem(0.2);
//                visualizer.setBackgroundColor(0.0,0.0,0.0);
//                visualizer.addPointCloud(model_clouds[2],
//                                         ColorHandlerTXYZ(model_clouds[2], 255.0, 0, 0.0),
//                                         "model_aligned_m");

//                while(!visualizer.wasStopped())
//                {
//                    visualizer.spinOnce();
//                }

                flann::Matrix<float> cvfh_data (new float[data_.histograms.size () * data_.histograms[0].size ()], data_.histograms.size (), data_.histograms[0].size ());
                for (size_t i = 0; i < data_.histograms.size(); ++i)
                {
                    cvfh_model cvfh_;
                    cvfh_.first = data_.ids[i];
                    for (size_t j = 0; j < data_.histograms[i].size(); ++j)
                    {
                        cvfh_data[i][j] = data_.histograms[i][j];
                    }
                    cvfh_.second = data_.histograms[i];
                    cvfh_models.push_back(cvfh_);

                }
                cvfh_index = new  flann::Index<flann::ChiSquareDistance<float> >(cvfh_data, flann::LinearIndexParams ());
                cvfh_index->buildIndex();
                data_loaded = true;
            }

        }

        Detection_experimental()
        {
            count = 0;

//            cloud_kinect_wo_plane = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
            cloud_temp = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
//            model_normals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);
            model_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
        }

        /** \brief Search for the closest k neighbors
        * \param index the tree
        * \param model the query model
        * \param k the number of neighbors to search for
        * \param indices the resultant neighbor indices
        * \param distances the resultant neighbor distances
        */
        inline void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
        {
            // Query point
            flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
            memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));
            indices = flann::Matrix<int>(new int[k], 1, k);
            distances = flann::Matrix<float>(new float[k], 1, k);
            index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
            delete[] p.ptr ();
        }

        static void declare_params(ecto::tendrils& params)
        {
            object_recognition_core::db::bases::declare_params_impl(params, "msd_pcl");

            params.declare(&Detection_experimental::quite_, "quite", "If true, not display msgs to user", false);
        }
        static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
        {

            inputs.declare<ecto::pcl::PointCloud>("input", "Input point cloud").required(true);
            outputs.declare(&Detection_experimental::pose_results_, "pose_results", "The results of object recognition");
        }

        void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
        {
            configure_impl();
            input_ = inputs["input"];
        }

        void process_input()
        {
            ecto::pcl::xyz_cloud_variant_t cv = input_->make_variant();
            __input_cloud = boost::get< boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > >(cv);

            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;

            seg.setOptimizeCoefficients (true);

            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setDistanceThreshold (0.01);

            seg.setInputCloud (__input_cloud);
            seg.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0)
            {
                std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp_ (new pcl::PointCloud<pcl::PointXYZRGB>);

            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud (__input_cloud);
            extract.setIndices (inliers);
            extract.setNegative (true);
            extract.filter (*cloud_temp);

            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];
            float d = coefficients->values[3];

            for (size_t i = 0; i < cloud_temp->points.size(); i++)
            {
                float plane = cloud_temp->points[i].x * a +
                              cloud_temp->points[i].y * b +
                              cloud_temp->points[i].z * c + d;
                if (c > 0)
                {
                    if (plane < 0)
                    {
                        cloud_temp_->points.push_back(cloud_temp->points[i]);
                    }
                }
                else
                {
                    if (plane > 0)
                    {
                        cloud_temp_->points.push_back(cloud_temp->points[i]);
                    }
                }
            }
            cloud_temp->clear();


            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
            tree->setInputCloud (cloud_temp_);
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
            ec.setClusterTolerance (0.02); // 2cm
            ec.setMinClusterSize (100);
            ec.setMaxClusterSize (25000);
            ec.setSearchMethod (tree);
            ec.setInputCloud (cloud_temp_);
            ec.extract (cluster_indices);

            clusters.clear();

            pcl::VoxelGrid<pcl::PointXYZRGB> sor;

            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
            {
                PCLCluster cluster;
                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
                    cluster.input_cloud->points.push_back (cloud_temp_->points[*pit]); //*
                cluster.input_cloud->width = cluster.input_cloud->points.size ();
                cluster.input_cloud->height = 1;
                cluster.input_cloud->is_dense = true;

                pcl::PointXYZRGB minpoint, maxpoint;
                pcl::getMinMax3D(*cluster.input_cloud, minpoint, maxpoint);

                if (fabs(minpoint.data[0] - maxpoint.data[0]) > 0.4 ||
                    fabs(minpoint.data[1] - maxpoint.data[1]) > 0.4 ||
                    fabs(minpoint.data[2] - maxpoint.data[2]) > 0.4)
                {
                    continue;
                }

                sor.setInputCloud (cluster.input_cloud);
                sor.setLeafSize (0.001f, 0.001f, 0.001f);
                sor.filter (*cluster.voxeled_cloud);

                pcl::search::KdTree<pcl::PointXYZRGB>::Ptr mls_tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
                pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
                mls.setComputeNormals (false);
                mls.setPolynomialFit (true);
                mls.setSearchMethod (mls_tree);
                mls.setSearchRadius (0.005);
                mls.setInputCloud (cluster.voxeled_cloud);
                mls.process (*cluster.smoothed_cloud);

                for (size_t i = 0; i < cluster.smoothed_cloud->size(); i++)
                {
                    if (isnan(cluster.smoothed_cloud->points[i].x))
                    {
                        cluster.smoothed_cloud->points[i].x = cluster.smoothed_cloud->points[i - 1].x;
                        cluster.smoothed_cloud->points[i].y = cluster.smoothed_cloud->points[i - 1].y;
                        cluster.smoothed_cloud->points[i].z = cluster.smoothed_cloud->points[i - 1].z;
                    }
                }
                pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
                norm_est.setRadiusSearch(0.005);
                norm_est.setInputCloud (cluster.smoothed_cloud);
                norm_est.compute (*cluster.normals);

                clusters.push_back(cluster);
            }

            std::cout << "Number of clusters: " << clusters.size() << std::endl;

//            pcl::visualization::PCLVisualizer visualizer("Visualiser");
//            visualizer.addCoordinateSystem(0.2);
//            visualizer.setBackgroundColor(0.0,0.0,0.0);
//            for (uint cluster_id = 0; cluster_id < clusters.size(); cluster_id++)
//            {
//                std::stringstream ss;
//                ss << cluster_id;
//                visualizer.addPointCloud(clusters[cluster_id].smoothed_cloud,
//                                         ColorHandlerTXYZRGB(clusters[cluster_id].smoothed_cloud, 255.0 - cluster_id * 50, cluster_id * 50, 0.0),
//                                         ss.str());
//            }
//            while(!visualizer.wasStopped())
//            {
//                visualizer.spinOnce();
//            }


            compute_cvfh();
        }

        void compute_cvfh()
        {
            pcl::OURCVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> cvfh_estimator;

            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cvfh_tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
            cvfh_estimator.setSearchMethod (cvfh_tree);
            // Compute the features
            cvfh_estimator.setEPSAngleThreshold(0.13f); // 5 degrees.
            cvfh_estimator.setCurvatureThreshold(0.025f);
            cvfh_estimator.setClusterTolerance (0.015f);
            cvfh_estimator.setNormalizeBins(false);
            cvfh_estimator.setAxisRatio(0.8);

            for (uint cluster_id = 0; cluster_id < clusters.size(); cluster_id++)
            {
                pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_signature (new pcl::PointCloud<pcl::VFHSignature308> ());
                cvfh_estimator.setInputCloud (clusters[cluster_id].smoothed_cloud);
                cvfh_estimator.setInputNormals (clusters[cluster_id].normals);
                cvfh_estimator.compute (*cvfh_signature);
                cvfh_estimator.getTransforms(clusters[cluster_id].transformations);
                cvfh_estimator.getValidTransformsVec(clusters[cluster_id].valid_trans);
                std::cout << "Signatures count: " << cvfh_signature->points.size() << std::endl;

                for (uint hist_id = 0; hist_id < cvfh_signature->points.size(); hist_id++)
                {
                    cvfh_model _cvfh_model;
                    std::stringstream cvfh_ss;
                    cvfh_ss << hist_id;
                    _cvfh_model.first = cvfh_ss.str();
                    _cvfh_model.second.resize(308);

                    for (size_t i = 0; i < 308; ++i)
                    {
                        _cvfh_model.second[i] = cvfh_signature->points[hist_id].histogram[i];
                    }
                    clusters[cluster_id].cvfh_models.push_back(_cvfh_model);
                }
            }
        }
        int process(const tendrils& inputs, const tendrils& outputs)
        {
            if (!data_loaded)
            {
                return ecto::QUIT;
            }
            process_input();
// Seraching for NN
            pose_results_->clear();

            for (uint cluster_id = 0; cluster_id < clusters.size(); cluster_id++)
            {
                std::vector<index_score> indices_scores;
                flann::Matrix<int> cvfh_k_indices;
                flann::Matrix<float> cvfh_k_distances;
                int k = 15;
                for (uint i = 0; i < clusters[cluster_id].cvfh_models.size(); i++)
                {
                    cvfh_model cvfh_nn;
                    cvfh_nn = clusters[cluster_id].cvfh_models[i];
                    nearestKSearch (*cvfh_index, cvfh_nn, k, cvfh_k_indices, cvfh_k_distances);

                    double score = 0;
                    for (int NN_ = 0; NN_ < k; ++NN_)
                    {
                        score = cvfh_k_distances[0][NN_];
                        index_score is;
                        is.idx_models_ = cvfh_k_indices[0][NN_];
                        is.idx_input_ = static_cast<int> (i);
                        is.score_ = score;
                        indices_scores.push_back (is);
                    }
                }
                std::sort (indices_scores.begin (), indices_scores.end (), sortIndexScoresOp);
       // Preparing some data (views, poses, etc ...)
                std::cout << "Number of hypotheses: " << indices_scores.size() << "\n";
                transforms_.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);
                descriptor_distances_.clear ();

                for (uint hyp_num = 0; hyp_num < indices_scores.size(); hyp_num++)
                {
                    std::string id_string = cvfh_models.at (indices_scores[hyp_num].idx_models_).first;
                    int view_id = atoi(id_string.substr(0, id_string.find_first_of("_")).c_str());
                    int idx_input = indices_scores[hyp_num].idx_input_;

                    Eigen::Matrix4f roll_view_pose;

                    roll_view_pose = data_.roll_transforms[indices_scores[hyp_num].idx_models_];

                    if(clusters[cluster_id].valid_trans[idx_input])
                    {

                        Eigen::Matrix4f model_view_pose;
                        model_view_pose = data_.poses[view_id];

                        Eigen::Matrix4f scale_mat;
                        scale_mat.setIdentity (4, 4);

                        Eigen::Matrix4f hom_from_OC_to_CC;
                        hom_from_OC_to_CC = scale_mat * clusters[cluster_id].transformations[idx_input].inverse () * roll_view_pose * model_view_pose;

                        transforms_->push_back (hom_from_OC_to_CC);
                        descriptor_distances_.push_back (static_cast<float> (indices_scores[hyp_num].score_));
                    }
                }
// Alighning
                pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_m (new pcl::PointCloud<pcl::PointXYZ>);

                scene_cloud_m->points.resize(clusters[cluster_id].smoothed_cloud->size());
                for (size_t i = 0; i < clusters[cluster_id].smoothed_cloud->points.size(); i++)
                {
                    scene_cloud_m->points[i].x = clusters[cluster_id].smoothed_cloud->points[i].x;
                    scene_cloud_m->points[i].y = clusters[cluster_id].smoothed_cloud->points[i].y;
                    scene_cloud_m->points[i].z = clusters[cluster_id].smoothed_cloud->points[i].z;
                }

                float VOXEL_SIZE_ICP_ = 0.005;
                pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_scene;
                voxel_grid_scene.setInputCloud (scene_cloud_m);
                voxel_grid_scene.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
                voxel_grid_scene.filter (*scene_cloud);

                if (transforms_->size() == 0)
                {
                    continue;
                }
                std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> pre_aligned_models;
                pre_aligned_models.resize (transforms_->size());
                for (uint hyp_num = 0; hyp_num < transforms_->size(); hyp_num++)
                {
    //                std::cout << "Starting alighning of hypothese num: " << hyp_num + 1 << "  \n";
                    pcl::PointCloud<pcl::PointXYZ>::Ptr model_aligned (new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr model_aligned_m (new pcl::PointCloud<pcl::PointXYZ>);

                    pcl::transformPointCloud (*model_clouds[model_num[indices_scores[hyp_num].idx_models_]], *model_aligned_m, transforms_->at (hyp_num));

                    pre_aligned_models[hyp_num] = model_aligned_m;

                    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
                    voxel_grid_icp.setInputCloud (model_aligned_m);
                    voxel_grid_icp.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
                    voxel_grid_icp.filter (*model_aligned);

                    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> reg;
                    reg.setInputSource(model_aligned); //model
                    reg.setInputTarget (scene_cloud); //scene
                    reg.setMaximumIterations (50);
                    reg.setMaxCorrespondenceDistance (0.01);
                    reg.setTransformationEpsilon (1e-6);

                    pcl::PointCloud<pcl::PointXYZ>::Ptr output_ (new pcl::PointCloud<pcl::PointXYZ> ());
                    reg.align (*output_);

                    Eigen::Matrix4f icp_trans = reg.getFinalTransformation ();
                    transforms_->at (hyp_num) = icp_trans * transforms_->at (hyp_num);
                }


// Hypotheses verification
                std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> aligned_models;
                aligned_models.resize (transforms_->size());

                for (size_t i = 0; i < transforms_->size(); i++)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr model_aligned (new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::transformPointCloud (*model_clouds[model_num[indices_scores[i].idx_models_]], *model_aligned, transforms_->at (i));
                    aligned_models[i] = model_aligned;
                }

                std::vector<bool> mask_hv;
                pcl::GlobalHypothesesVerification<pcl::PointXYZ, pcl::PointXYZ>  hv_algorithm_;

                hv_algorithm_.setResolution (0.005f);
                hv_algorithm_.setMaxIterations (7000);
                hv_algorithm_.setInlierThreshold (0.008f);
                hv_algorithm_.setRadiusClutter (0.04f);
                hv_algorithm_.setRegularizer (3.f);
                hv_algorithm_.setClutterRegularizer (7.5f);
                hv_algorithm_.setDetectClutter (1);
                hv_algorithm_.setOcclusionThreshold (0.01f);
                hv_algorithm_.setSceneCloud (scene_cloud);
                hv_algorithm_.addModels (aligned_models, true);
                hv_algorithm_.verify ();
                hv_algorithm_.getMask (mask_hv);

                boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms_final;
                transforms_final.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

                bool pose_founded = false;
                int winner = 0;
                for (size_t i = 0; i < mask_hv.size(); i++)
                {
                    if (mask_hv[i])
                    {
                        std::cout << "Mask #" << i + 1 << " : " << (int)mask_hv[i] << "\n";
                        pose_founded = true;
                        transforms_final->push_back (transforms_->at (i));
                        winner = i;
                    }
                }
                if (!pose_founded)
                {
                    continue;
                };

//Publish pose

                cv::Matx33f final_mat;
                final_mat << transforms_final->at (0)(0,0), transforms_final->at (0)(0,1), transforms_final->at (0)(0,2),
                        transforms_final->at (0)(1,0), transforms_final->at (0)(1,1), transforms_final->at (0)(1,2),
                        transforms_final->at (0)(2,0), transforms_final->at (0)(2,1), transforms_final->at (0)(2,2);

                cv::Vec3f final_vec(transforms_final->at (0)(0,3), transforms_final->at (0)(1,3), transforms_final->at (0)(2,3));

                std::cout << "Final mat: " << final_mat << std::endl;
                std::cout << "Final vec: " << final_vec << std::endl;

    //            std::cout << "Publish topic? (0 - publish; 1 - no, find again; 2 - no, EXIT):\n";
    //            int command;
    //            std::cin >> command;

    //            switch(command)
    //            {
    //                case 0: break;
    //                case 1: return ecto::DO_OVER;
    //                case 2: return ecto::QUIT;
    //            }

                object_recognition_core::common::PoseResult pose_result;
                pose_result.set_R(cv::Mat(final_mat));
                pose_result.set_T(cv::Mat(final_vec));
                pose_result.set_object_id(db_, hist_object_ids[indices_scores[winner].idx_models_]);
                pose_results_->push_back(pose_result);
            }

            cloud_temp->clear();
            return ecto::OK;

        }

    public:
        cv::Vec3f T;
        cv::Matx33f R;
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_kinect_wo_plane;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp;
//        pcl::PointCloud<pcl::Normal>::Ptr model_normals;
        pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud;
        std::vector<typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr> model_clouds;
        boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > __input_cloud;

        ecto::spore<bool> quite_;
        ecto::spore<cv::Mat> R_, T_;
//        ecto::spore<ecto::pcl::FeatureCloud> features_;
        ecto::spore<ecto::pcl::PointCloud> input_;
        ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
//        std::vector<vfh_model> vfh_models;
        std::vector<cvfh_model> cvfh_models;
//        std::vector<cvfh_model> cvfh_models_input;

//        std::vector<bool> valid_trans;
//        std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms_;
        std::vector<float> descriptor_distances_;

//        cvfh_model _cvfh_model;
        flann::Index<flann::ChiSquareDistance<float> > *cvfh_index;

        std::vector<PCLCluster> clusters;

        int count;
        bool data_loaded;

        object_recognition_core::db::data data_;
        std::vector<std::string> poses_object_ids;
        std::vector<std::string> hist_object_ids;
        std::vector<int> model_num;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };//struct Detection
}

ECTO_CELL(ecto_msd_pcl_exp, ecto_msd_pcl::Detection_experimental, "Detector_experimental", "Detect the object by msd_pcl detection algorithm.")


//            int y_s = (int)floor (sqrt ((double)k));
//            int x_s = y_s + (int)ceil ((k / (double)y_s) - y_s);
//            double x_step = (double)(1 / (double)x_s);
//            double y_step = (double)(1 / (double)y_s);

//            int viewport = 0, l = 0, m = 0;
//            int cloud_count = 0;
//            pcl::visualization::PCLVisualizer p ("VFH Cluster Classifier");

//            for (size_t i = 0; i < transforms_->size(); ++i)
//            {
//                std::string cloud_name = data_.ids[i];
//                std::stringstream ss;
//                ss << "kinect_" << cloud_count;
//                std::stringstream ss2;
//                ss2 << "kinect11_" << cloud_count;
//                p.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
//                l++;
//                if (l >= x_s)
//                {
//                    l = 0;
//                    m++;
//                }

//                Eigen::Vector4f centroid;
//                pcl::compute3DCentroid (*aligned_models[i], centroid);
//                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_demean (new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prealigned_demean (new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_kinect_demean (new pcl::PointCloud<pcl::PointXYZRGB>);
//                pcl::demeanPointCloud<pcl::PointXYZ> (*aligned_models[i], centroid, *cloud_xyz_demean);
//                pcl::demeanPointCloud<pcl::PointXYZ> (*pre_aligned_models[i], centroid, *cloud_prealigned_demean);
//                pcl::demeanPointCloud<pcl::PointXYZRGB> (*cloud_kinect_wo_plane, centroid, *cloud_kinect_demean);

//                // Add to renderer*
//                p.addPointCloud(cloud_xyz_demean,ColorHandlerTXYZ(cloud_xyz_demean, 0.0, 255.0, 0.0), cloud_name, viewport);
//                p.addPointCloud(cloud_kinect_demean,ColorHandlerTXYZRGB(cloud_kinect_demean, 255.0, 0.0, 0.0), ss.str(), viewport);
//                p.addPointCloud(cloud_prealigned_demean,ColorHandlerTXYZ(cloud_prealigned_demean, 0.0, 0.0, 255.0), ss2.str(), viewport);
//                cloud_count++;

//            }

//            p.setSize(1920, 1080);

//            while (!p.wasStopped())
//            {
//                p.spinOnce();
//            }



// Visualize results
//            pcl::PointCloud<pcl::PointXYZ>::Ptr model_aligned (new pcl::PointCloud<pcl::PointXYZ>);
//            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_final->at (0));
//            pcl::visualization::PCLVisualizer visualizer("Visualiser");
//            visualizer.addCoordinateSystem(0.2);
//            visualizer.addPointCloud(scene_cloud,ColorHandlerTXYZ(scene_cloud, 255.0, 0.0, 0.0), "scene_cloud");
//            visualizer.addPointCloud(model_aligned,ColorHandlerTXYZ(model_aligned, 0.0, 255.0, 0.0), "model_aligned");
//            visualizer.addPointCloud(model_cloud,ColorHandlerTXYZ(model_cloud, 0.0, 0.0, 150.0), "model_cloud");
//            while(!visualizer.wasStopped())
//            {
//                visualizer.spinOnce();
//            }
//            visualizer.close();

//            std::cout << "What to do?  0 - Calculate pose; 1 - Try again; 2 - Exit; 3 - Enter variant." << std::endl;
//            int correct;

//            std::cin >> correct;

/*
//            pcl::console::print_highlight ("The closest %d neighbors are:\n", k * cvfh_models_input.size());
//            for (uint i = 0; i < k * cvfh_models_input.size(); ++i)
//            {
//                pcl::console::print_info ("    %d - %s (%d) with a distance of: %f\n",
//                                          i, cvfh_models.at (indices_scores[i].idx_models_).first.c_str (), indices_scores[i].idx_models_, indices_scores[i].score_);

//            }

            Eigen::Matrix4f transformation_matrix;

            for (int m = 0; m < 4; m++)
            {
                for (int n = 0; n < 4; n++)
                {
                    transformation_matrix(m,n) = atof(closest_neighbor_vfh.substr(0, closest_neighbor_vfh.find_first_of(",")).c_str());
                    closest_neighbor_vfh = closest_neighbor_vfh.substr(closest_neighbor_vfh.find_first_of(",") + 1, closest_neighbor_vfh.length() - 1);
                }
            }

            float length = sqrt(transformation_matrix(0,3)*transformation_matrix(0,3)
                                + transformation_matrix(1,3)*transformation_matrix(1,3)
                                + transformation_matrix(2,3)*transformation_matrix(2,3));
            std::cout << "transformation_matrix:" << std::endl;
            std::cout << transformation_matrix<< std::endl;

            std::cout << "radius:" << std::endl;
            std::cout << length << std::endl;

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid (*cloud_kinect_wo_plane, centroid);

            if(!*quite_)
            {
                std::cout << "centroid: " << centroid << std::endl;
            }
*/

//            pcl::PointCloud<pcl::PointXYZ>::Ptr loaded_cloud_Ptr(new pcl::PointCloud<pcl::PointXYZ>);
//            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > loaded_clouds;

//            for (int i = 0; i < k; ++i)
//            {
//                pcl::console::print_info ("    %d - %s (%d) with a distance of: %f\n", i, vfh_models.at (vfh_k_indices[0][i]).first.c_str (), vfh_k_indices[0][i], vfh_k_distances[0][i]);
//                std::stringstream ss;
//                ss << vfh_k_indices[0][i] << ".pcd";

//                std::string xyzrgb_adress = ss.str();
//                std::cout << xyzrgb_adress.c_str() << "\n";
//                pcl::PointCloud<pcl::PointXYZ>::Ptr loaded_cloud_Ptr(new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::io::loadPCDFile(xyzrgb_adress.c_str(), *loaded_cloud_Ptr);
//                loaded_clouds.push_back(loaded_cloud_Ptr);

//                std::cout << "ended \n";
//            }

/*

                std::stringstream ss;
                ss << "cloud_";
                if (chosen_index < 10)
                {
                    ss << "000" << chosen_index << ".pcd";
                }
                else if (chosen_index >= 10)
                {
                    ss << "00" << chosen_index << ".pcd";
                }
                std::string xyzrgb_adress = ss.str();

                if(!*quite_)
                {
                    std::cout << xyzrgb_adress.c_str() << std::endl;
                }

                //loading xyzrgb pointcloud

                pcl::io::loadPCDFile(xyzrgb_adress.c_str(), *loaded_cloud_Ptr);

                Eigen::Vector4f centroid;
                pcl::compute3DCentroid (*__input_cloud, centroid);

                if(!*quite_)
                {
                    std::cout << "centroid: " << centroid << std::endl;
                }

                cv::Mat T_mat = *T_;
                cv::Mat R_mat = *R_;

                if (T_mat.rows == 3 && R_mat.rows == 3)
                {
                     T = *T_;
                     R = *R_;
                }
                Eigen::Matrix4f translation_matrix;
                Eigen::Matrix4f rotation_matrix;
//                rotation_matrix << R(0,0), R(0,1), R(0,2), 0,
//                                        R(1,0), R(1,1), R(1,2), 0,
//                                        R(2,0), R(2,1), R(2,2), 0,
//                                        0, 0, 0, 1;
                rotation_matrix << R(0,0), R(1,0), R(2,0), 0,
                                        R(0,1), R(1,1), R(2,1), 0,
                                        R(0,2), R(1,2), R(2,2), 0,
                                        0, 0, 0, 1;
                translation_matrix << 1, 0, 0, -T(0),
                                        0, 1, 0, -T(1),
                                        0, 0, 1, -T(2),
                                        0, 0, 0, 1;
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr loaded_transformed_cloud_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_transformed_cloud_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::transformPointCloud(*loaded_cloud_Ptr, *loaded_transformed_cloud_Ptr, translation_matrix);
                pcl::transformPointCloud(*loaded_transformed_cloud_Ptr, *loaded_transformed_cloud_Ptr, rotation_matrix);
                pcl::transformPointCloud(*__input_cloud, *input_transformed_cloud_Ptr, translation_matrix);
                pcl::transformPointCloud(*input_transformed_cloud_Ptr, *input_transformed_cloud_Ptr, rotation_matrix);

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud_Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
                icp.setMaximumIterations(1000);
                icp.setInputSource(loaded_transformed_cloud_Ptr);
                icp.setInputTarget(input_transformed_cloud_Ptr);
                icp.setMaxCorrespondenceDistance(0.3);
                //icp.setEuclideanFitnessEpsilon(1);

                if(!*quite_)
                {
                    std::cout <<"Starting icp alignment" << std::endl;
                }

                icp.align(*aligned_cloud_Ptr);

                if(!*quite_)
                {
                    std::cout <<"Alignment ended: " << icp.hasConverged() << std::endl;
                }

                pose_results_->clear();

                PoseResult pose_result;

                Eigen::Matrix4f icp_transform = icp.getFinalTransformation();

                cv::Matx33f icp_mat, rotZ, rotZ_add, final_mat;

                icp_mat << icp_transform(0,0), icp_transform(0,1), icp_transform(0,2),
                        icp_transform(1,0), icp_transform(1,1), icp_transform(1,2),
                        icp_transform(2,0), icp_transform(2,1), icp_transform(2,2);

                float g = (-chosen_index * 6) * 3.1415926 / 180;

                rotZ << cos(g), -sin(g), 0,
                       sin(g),  cos(g), 0,
                       0,            0, 1;

                rotZ_add << cos(g), -sin(g), 0,
                       sin(g),  cos(g), 0,
                       0,            0, 1;

                final_mat = R * rotZ * icp_mat;

                cv::Vec3f icp_vec(icp_transform(0,3) + 0.01, icp_transform(1,3), icp_transform(2,3) - 0.005);
                cv::Vec3f final_vec = T + R * icp_vec;

                cv::Mat RR = cv::Mat(final_mat);

                pose_result.set_R(RR);
                pose_result.set_T(cv::Mat(final_vec));
                pose_result.set_object_id(db_, object_ids_.front());

                pose_results_->push_back(pose_result);
                finded_count++;

                if(!*quite_)
                {
                    std::cout << "T: " << T << std::endl;
                    std::cout << "R: " << R << std::endl;
                    std::cout << "Icp vector: " << icp_vec << std::endl;
                    std::cout << "Icp matrix: " << icp_mat << std::endl;
                    std::cout << "Final vector: " << final_vec << std::endl;
                    std::cout << "Final matrix: " << RR << std::endl;
                }
*/
/*
                pcl::visualization::PCLVisualizer visualizer("Visualiser");
                visualizer.addCoordinateSystem(0.2);
                visualizer.addPointCloud(__input_cloud,ColorHandlerTXYZ(__input_cloud, 122.0, 0.0, 0.0), "input_cloud_Ptr");
                visualizer.addPointCloud(loaded_cloud_Ptr,ColorHandlerTXYZ(loaded_cloud_Ptr, 0.0, 0.0, 122.0), "loaded_cloud_Ptr");
                visualizer.addPointCloud(loaded_transformed_cloud_Ptr,ColorHandlerTXYZ(loaded_transformed_cloud_Ptr, 0.0, 0.0, 255.0), "loaded_transformed_cloud_Ptr");
                visualizer.addPointCloud(input_transformed_cloud_Ptr,ColorHandlerTXYZ(input_transformed_cloud_Ptr, 255.0, 0.0, 0.0), "input_transformed_cloud_Ptr");
                visualizer.addPointCloud(aligned_cloud_Ptr,ColorHandlerTXYZ(aligned_cloud_Ptr, 0.0, 255.0, 0.0), "aligned_cloud_Ptr");
                while(!visualizer.wasStopped())
                {
                    visualizer.spinOnce();
                }
                visualizer.close();*/


/*pcl::demeanPointCloud<pcl::PointXYZRGB> (*__input_cloud, centroid, *input_demeaned_cloud_Ptr);
pcl::compute3DCentroid (*loaded_cloud_Ptr, centroid);
pcl::demeanPointCloud<pcl::PointXYZRGB> (*loaded_cloud_Ptr, centroid, *loaded_demeaned_cloud_Ptr);

pcl::visualization::PCLVisualizer visualizer_dem("Visualiser");
visualizer_dem.addCoordinateSystem(0.2);

visualizer_dem.addPointCloud(input_demeaned_cloud_Ptr,ColorHandlerTXYZ(input_demeaned_cloud_Ptr, 0.0, 255.0, 0.0), "input_demeaned_cloud_Ptr");
visualizer_dem.addPointCloud(loaded_demeaned_cloud_Ptr,ColorHandlerTXYZ(loaded_demeaned_cloud_Ptr, 255.0, 0.0, 0.0), "loaded_demeaned_cloud_Ptr");
while(!visualizer_dem.wasStopped())
{
    visualizer_dem.spinOnce();
}
visualizer_dem.close();*/


