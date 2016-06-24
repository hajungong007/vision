#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

#include <ecto/ecto.hpp>
#include <ecto_pcl/ecto_pcl.hpp>
#include <ecto_pcl/pcl_cell.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/apps/render_views_tesselated_sphere.h>
#include <pcl/features/crh.h>

#include <pcl/recognition/hv/hv_go.h>

#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>

#include <boost/filesystem.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>

#include "db_msd_pcl.h"
#include "persistence_utils.h"
#include "vtk_model_sampling.h"

using ecto::tendrils;
using ecto::spore;
typedef std::pair<std::string, std::vector<float> > vfh_model;
typedef std::pair<std::string, std::vector<float> > cvfh_model;

typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerTXYZ;
using object_recognition_core::db::Document;
using object_recognition_core::db::DocumentId;


namespace ecto_msd_pcl
{
    struct CAD_trainer
    {

        CAD_trainer()
        {

        }
        void CreateDetectionConf()
        {
            std::string prefix = "/home/msdu/catkin_ws/src/vision/pcl/conf/detection.";
            //*model_path // "/home/msd-u64/catkin_ws/data/test/domik_v_derevne.ply";
            std::string _model_path = *model_path;

            std::size_t first_point = _model_path .find_last_of("/");
            std::size_t second_point = _model_path .find_last_of(".");

            std::string name = _model_path.substr(first_point + 1,second_point - first_point - 1);
            std::cout << name << std::endl;
            std::string filename = prefix + name;
            ofstream myfile (filename.c_str());
            if (myfile.is_open())
            {
                myfile << "sink1:" << std::endl;
                myfile << "  type: Publisher" << std::endl;
                myfile << "  module: 'object_recognition_ros.io'" << std::endl;
                myfile << "  inputs: [pipeline1]" << std::endl;
                myfile << "pipeline1:" << std::endl;
                myfile << "  type: MsdPclDetector" << std::endl;
                myfile << "  module: 'object_recognition_msd_pcl'" << std::endl;
                myfile << "  outputs: [sink1]" << std::endl;
                myfile << "  parameters:" << std::endl;
                myfile << "    object_ids: ['"<< *object_id_in << "'] " << std::endl;
                myfile << "    sensor: 'kinect'" << std::endl;
                myfile << "    db:" << std::endl;
                myfile << "      type: 'CouchDB'" << std::endl;
                myfile << "      root: 'http://localhost:5984'" << std::endl;
                myfile << "      collection: 'object_recognition'" << std::endl;
                myfile.close();
            }

        }
        static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
        {
            outputs.declare(&CAD_trainer::json_db_out, "json_db", "Database parameters");
            outputs.declare(&CAD_trainer::object_id_out, "object_id", "The object id, to associate this model with.");
            outputs.declare(&CAD_trainer::db_document_, "db_document", "The filled document.");
            outputs.declare(&CAD_trainer::commit_, "commit", "Upload to db.");

        }
        static void declare_params(ecto::tendrils& params)
        {
            params.declare(&CAD_trainer::json_db_in, "json_db_in", "The DB parameters", "{}").required(true);
            params.declare(&CAD_trainer::object_id_in, "object_id_in", "The object id, to associate this model with.", "{}").required(true);
            params.declare(&CAD_trainer::model_path, "model_path", "Path to objects mesh", "{}").required(true);
            params.declare(&CAD_trainer::commit, "commit", "commit", "{}").required(true);
        }
        void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
        {

        }

        void generate_views()
        {
            CreateDetectionConf();
            std::string mesh_path = *model_path;
            vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New ();
            reader->SetFileName (mesh_path.c_str ());
            vtkSmartPointer < vtkPolyDataMapper > mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
            mapper->SetInputConnection (reader->GetOutputPort ());
            mapper->Update ();
            vtkSmartPointer<vtkPolyData> vtkPolyData_ = mapper->GetInput();

            pcl::apps::RenderViewsTesselatedSphere render_views;
            render_views.addModelFromPolyData (vtkPolyData_);
            render_views.setTesselationLevel(2);

            render_views.generateViews ();
            render_views.getViews(views);
            render_views.getPoses(poses);

            model_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::rec_3d_framework::uniform_sampling (vtkPolyData_, 100000, *model_cloud);
            std::cout << "Views: " << views.size() << std::endl;
        }
        void MLSFIlter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out)
	{
	    pcl::search::KdTree<pcl::PointXYZ>::Ptr Tree (new pcl::search::KdTree<pcl::PointXYZ>);
	    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> MLSFIlter;
	    MLSFIlter.setComputeNormals (false);
	    MLSFIlter.setPolynomialFit (true);
	    MLSFIlter.setSearchMethod (Tree);
	    MLSFIlter.setSearchRadius (0.015);
	    MLSFIlter.setInputCloud (cloud_in);
	    MLSFIlter.process (*cloud_out);
	    
	    RemoveNANs(cloud_out);

	}
	void RemoveNANs(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
	{
	    for (size_t i = 0; i < cloud_in->size(); i++)
	    {
		if (isnan(cloud_in->points[i].x))
		{
		    cloud_in->points[i].x = cloud_in->points[i - 1].x;
		    cloud_in->points[i].y = cloud_in->points[i - 1].y;
		    cloud_in->points[i].z = cloud_in->points[i - 1].z;
		}
	    }
	}
        void process_pointclouds()
        {
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses_;
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> views_;
            
            for (size_t i = 0; i < views.size(); i++)
            {

                views_.push_back(views[i]);
                poses_.push_back(poses[i]);

            }
            std::cout << "Views: " << views.size() << "\n";
            std::cout << "Views_: " << views_.size() << "\n";
            std::cout << "poses: " << poses.size() << "\n";
            std::cout << "poses_: " << poses_.size() << "\n";

	    std::string hist_filename = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/hists.txt";
	    std::string rolls_filename = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/rolls.txt";
	    std::string poses_filename = "/home/msdu/catkin_ws/src/vision/pcl/test/ourcvfh_regression/data/poses.txt";
            ofstream hists_file (hist_filename.c_str());
	    ofstream rolls_file (rolls_filename.c_str());
	    ofstream poses_file (poses_filename.c_str());
// 	    if (myfile.is_open())
// 	    {
// 		myfile << "num_views:" << views.size() << std::endl;
// 	    }

            for (size_t i = 0; i < views_.size(); i++)
// 	    for (size_t i = 0; i < 43; i++)
            {
		float VOXEL_LEAF_SIZE = 0.005;
		pcl::VoxelGrid<pcl::PointXYZ> VoxelGridFIlter;
		VoxelGridFIlter.setInputCloud (views[i]);
		VoxelGridFIlter.setLeafSize (VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE);
		VoxelGridFIlter.filter (*views_[i]);
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr view (new pcl::PointCloud<pcl::PointXYZ>);
		
		MLSFIlter(views_[i], view);	
		
		std::cout << "Start pcl processing #" << i << std::endl;
                pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal> ());
                pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
                norm_est.setRadiusSearch(0.020);
                norm_est.setInputCloud (view);
                norm_est.compute (*model_normals);
//CVFH computing

                pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh_estimator;
                cvfh_estimator.setInputCloud (view);
                cvfh_estimator.setInputNormals (model_normals);
                pcl::search::KdTree<pcl::PointXYZ>::Ptr cvfh_tree (new pcl::search::KdTree<pcl::PointXYZ> ());
                cvfh_estimator.setSearchMethod (cvfh_tree);
                pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_signature (new pcl::PointCloud<pcl::VFHSignature308> ());
                // Compute the features

                cvfh_estimator.setEPSAngleThreshold(0.13f); // 5 degrees.
                cvfh_estimator.setCurvatureThreshold(0.025f);
                cvfh_estimator.setClusterTolerance (0.015f);
                cvfh_estimator.setNormalizeBins(false);
                cvfh_estimator.setAxisRatio(0.8);
		try
		{
		    cvfh_estimator.compute (*cvfh_signature);
		}
		catch(...)
		{
		    continue;
		}
		std::cout << "Signatures count: " << cvfh_signature->points.size() << std::endl;


                std::vector < Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>  > centroids;
                std::vector<bool> valid_trans;
                std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms;

                cvfh_estimator.getCentroidClusters(centroids);
                cvfh_estimator.getTransforms(transforms);
                cvfh_estimator.getValidTransformsVec(valid_trans);

                int view_id = i;

                data_.poses.push_back(poses_[view_id]);

                for (uint hist_id = 0; hist_id < cvfh_signature->points.size(); hist_id++)
                {
                    std::stringstream cvfh_ss;

                    cvfh_ss << view_id << "," << hist_id;

                    cvfh_model cvfh;
                    cvfh.first = cvfh_ss.str();
                    cvfh.second.resize(308);
                    for (int j = 0; j < 308; j++)
                    {
                        cvfh.second[j] = cvfh_signature->points[hist_id].histogram[j];
                    }
                    cvfh_models.push_back(cvfh);

                    std::stringstream id_ss;
                    id_ss << view_id << "_" << hist_id;
                    data_.ids.push_back(id_ss.str());
                    data_.centroids.push_back(centroids[hist_id]);
                    data_.roll_transforms.push_back(transforms[hist_id]);
                    data_.histograms.push_back(cvfh.second);
		    if (hists_file.is_open())
		    {
// 			myfile << "view_" << i << "_hist_" << hist_id << ":" << cvfh.second << std::endl;
// 			myfile << "view_" << i << "_roll_" << hist_id << ":" << transforms[hist_id] << std::endl;
// 			myfile << "view_" << i << "_pose_" << hist_id << ":" << poses_[view_id] << std::endl;

// 			hists_file << "view_" << i << "_hist_" << hist_id << ": [";
			for (size_t j = 0; j < cvfh.second.size(); j++)
			{
			    hists_file << cvfh.second[j] << " ";
			}
			hists_file << "\n";
		    }
		    if (rolls_file.is_open())
		    {
			rolls_file << transforms[hist_id](0,0) << " " << transforms[hist_id](0,1) << " " << transforms[hist_id](0,2) << " " << transforms[hist_id](0,3) << " "
				<< transforms[hist_id](1,0) << " " << transforms[hist_id](1,1) << " " << transforms[hist_id](1,2) << " " << transforms[hist_id](1,3) << " "
				<< transforms[hist_id](2,0) << " " << transforms[hist_id](2,1) << " " << transforms[hist_id](2,2) << " " << transforms[hist_id](2,3) << " "
				<< transforms[hist_id](3,0) << " " << transforms[hist_id](3,1) << " " << transforms[hist_id](3,2) << " " << transforms[hist_id](3,3) << "\n";
		    }
		    if (poses_file.is_open())
		    {
			poses_file << poses_[view_id](0,0) << " " << poses_[view_id](0,1) << " " << poses_[view_id](0,2) << " " << poses_[view_id](0,3) << " "
				<< poses_[view_id](1,0) << " " << poses_[view_id](1,1) << " " << poses_[view_id](1,2) << " " << poses_[view_id](1,3) << " "
				<< poses_[view_id](2,0) << " " << poses_[view_id](2,1) << " " << poses_[view_id](2,2) << " " << poses_[view_id](2,3) << " "
				<< poses_[view_id](3,0) << " " << poses_[view_id](3,1) << " " << poses_[view_id](3,2) << " " << poses_[view_id](3,3) << "\n";			
		    }
                }
            }
            hists_file.close();
	    rolls_file.close();
	    poses_file.close();
            std::cout << "cvfh_models: " << cvfh_models.size() << std::endl;
        }

        int process(const tendrils& inputs, const tendrils& outputs)
        {
            generate_views();
            process_pointclouds();

            db_document.set_attachment<object_recognition_core::db::data > ("data", data_); // histograms
            db_document.set_attachment<pcl::PointCloud<pcl::PointXYZ> > ("model_cloud", *model_cloud);
            *db_document_ = db_document;
            *json_db_out = *json_db_in;
            *object_id_out = *object_id_in;
            *commit_ = *commit;

//            pcl::visualization::PCLVisualizer visualizer("Visualiser");
//            visualizer.addCoordinateSystem(0.2);
//            visualizer.addPointCloud(views[0],ColorHandlerTXYZ(views[0], 0.0, 0.0, 255.0), "views[0]");
//            visualizer.spin();

            return ecto::OK;
        }
        object_recognition_core::db::ObjectDbPtr db;
        ecto::spore<bool> commit_;
        ecto::spore<bool> commit;
        ecto::spore<std::string> json_db_out;
        ecto::spore<std::string> json_db_in;
        ecto::spore<Document> db_document_;
        ecto::spore<DocumentId> object_id_out;
        ecto::spore<DocumentId> object_id_in;
        ecto::spore<std::string> model_path;

        Document db_document;

        std::vector<cvfh_model> cvfh_models;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> views;

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud;

        object_recognition_core::db::data data_;

    };//struct Trainer
}

ECTO_CELL(ecto_msd_pcl_exp, ecto_msd_pcl::CAD_trainer, "CAD_trainer", "CAD_trainer the msd_pcl object detection algorithm.")
