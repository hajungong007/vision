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
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>


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



using ecto::tendrils;
using ecto::spore;
typedef std::pair<std::string, std::vector<float> > vfh_model;
typedef std::pair<std::string, std::vector<float> > esf_model;
typedef std::pair<std::string, std::vector<float> > cvfh_model;

namespace ecto_msd_pcl
{
	struct Trainer
	{
        Trainer()
        {
            count = 0;
        }
        static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
		{
            inputs.declare<cv::Mat>("R", "The orientation.").required(true);
            inputs.declare<cv::Mat>("T", "The translation.").required(true);
            inputs.declare<ecto::pcl::FeatureCloud>("features", "View point features");
            inputs.declare<ecto::pcl::PointCloud>("input", "Input point cloud");
            inputs.declare<bool>("last", "True if this is a last frame");
            inputs.declare<bool>("novel", "True if frame is new");
		}

		void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
		{
            R_ = inputs["R"];
            T_ = inputs["T"];
            features_ = inputs["features"];
            input_ = inputs["input"];
            last_ = inputs["last"];
            novel_ = inputs["novel"];
		}

        int process(const tendrils& inputs, const tendrils& outputs)
		{
            if (*novel_)
            {
                std::string kdtree_vfh_idx_file_name = "kdtree_vfh.idx";
                std::string training_vfh_data_h5_file_name = "training_vfh_data.h5";
                std::string training_vfh_data_list_file_name = "training_vfh_data.list";

                std::string kdtree_esf_idx_file_name = "kdtree_esf.idx";
                std::string training_esf_data_h5_file_name = "training_esf_data.h5";
                std::string training_esf_data_list_file_name = "training_esf_data.list";

                std::string kdtree_cvfh_idx_file_name = "kdtree_cvfh.idx";
                std::string training_cvfh_data_h5_file_name = "training_cvfh_data.h5";
                std::string training_cvfh_data_list_file_name = "training_cvfh_data.list";

                pcl::PointCloud<pcl::PointXYZRGB> input_cloud;
                pcl::PointCloud<pcl::VFHSignature308> features;

                // convert ecto::tendrils to pcl::clouds
                ecto::pcl::xyz_cloud_variant_t cv = input_->make_variant();
                boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > __input_cloud = boost::get< boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > >(cv);
                input_cloud = *__input_cloud; //xyzrgb pointcloud

                ecto::pcl::feature_cloud_variant_t fv = features_->make_variant();
                boost::shared_ptr<const pcl::PointCloud< pcl::VFHSignature308> > __features = boost::get< boost::shared_ptr<const pcl::PointCloud< pcl::VFHSignature308> > >(fv);
                features = *__features; // vfh features

                //Testing SHOTEstimation

                pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal> ());
                pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
                norm_est.setRadiusSearch(0.005);
                norm_est.setInputCloud (__input_cloud);
                norm_est.compute (*model_normals);

                pcl::VFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> vfh_estimator;
                vfh_estimator.setInputCloud (__input_cloud);
                vfh_estimator.setInputNormals (model_normals);
                pcl::search::KdTree<pcl::PointXYZRGB>::Ptr vfh_tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
                vfh_estimator.setSearchMethod (vfh_tree);
                pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_signature (new pcl::PointCloud<pcl::VFHSignature308> ());
                // Compute the features
                vfh_estimator.compute (*vfh_signature);



                pcl::CVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> cvfh_estimator;
                pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cvfh_tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
                pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_signature (new pcl::PointCloud<pcl::VFHSignature308> ());
                cvfh_estimator.setSearchMethod (cvfh_tree);
                cvfh_estimator.setInputCloud (__input_cloud);
                cvfh_estimator.setInputNormals (model_normals);
                cvfh_estimator.setEPSAngleThreshold(12);
                //cvfh_estimator.setCurvatureThreshold(max_curv);
                cvfh_estimator.setNormalizeBins(false);
                cvfh_estimator.compute (*cvfh_signature);

/*
                std::stringstream mls_ss;
                mls_ss << count << ".mls.pcd";
                pcl::io::savePCDFileASCII(mls_ss.str(), mls_points);





                pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
                pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> descr_est;
                descr_est.setRadiusSearch (0.01);
                descr_est.setInputCloud (__input_cloud);
                descr_est.setInputNormals (model_normals);
                descr_est.compute (*model_descriptors);

                std::stringstream ss2;
                ss2 << count << ".SHOT.pcd";
                pcl::io::savePCDFileASCII(ss2.str(), *model_descriptors);

                pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
                fpfh.setInputCloud (__input_cloud);
                fpfh.setInputNormals (model_normals);
                pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
                fpfh.setSearchMethod (tree);
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
                fpfh.setRadiusSearch (0.01);
                fpfh.compute (*fpfhs);

                std::stringstream ss3;
                ss3 << count << ".FPFH.pcd";
                pcl::io::savePCDFileASCII(ss3.str(), *fpfhs);

*/
                pcl::ESFEstimation<pcl::PointXYZRGB> esf_estimator;
                pcl::PointCloud<pcl::ESFSignature640>::Ptr esf_signature (new pcl::PointCloud<pcl::ESFSignature640> ());
                esf_estimator.setInputCloud (__input_cloud);
                esf_estimator.compute (*esf_signature);

                //end of testing

                std::stringstream vfh_ss;
                vfh_ss << count << ".vfh_signature.pcd";
                //pcl::io::savePCDFileASCII(vfh_ss.str(), features);

                std::stringstream esf_ss;
                esf_ss << count << ".esf_signature.pcd";
                //pcl::io::savePCDFileASCII(esf_ss.str(), features);

                std::stringstream cvfh_ss;
                cvfh_ss << count << ".cvfh_signature.pcd";
                //pcl::io::savePCDFileASCII(esf_ss.str(), features);

                vfh_model vfh;
                cvfh_model cvfh;
                esf_model esf;

                vfh.second.resize(308);
                vfh.first = vfh_ss.str();
                for (int i = 0; i < 308; i++)
                {
                    vfh.second[i] = vfh_signature->points[0].histogram[i];
                    //esf.second[i] = esf_signature->points[0].histogram[i];
                }
                vfh_models.push_back(vfh);

                cvfh.second.resize(308);
                cvfh.first = cvfh_ss.str();
                for (int i = 0; i < 308; i++)
                {
                    cvfh.second[i] = cvfh_signature->points[0].histogram[i];
                    //esf.second[i] = esf_signature->points[0].histogram[i];
                }
                cvfh_models.push_back(cvfh);

                esf.second.resize(640);
                esf.first = esf_ss.str();

                for (int i = 0; i < 640; i++)
                {
                    //vfh.second[i] = features.points[0].histogram[i];
                    esf.second[i] = esf_signature->points[0].histogram[i];
                }

                esf_models.push_back(esf);

                if (*last_)
                {
                    std::cout << "------------------ Last! --------------------" << std::endl;
                    flann::Matrix<float> vfh_data (new float[vfh_models.size () * vfh_models[0].second.size ()], vfh_models.size (), vfh_models[0].second.size ());

                    for (size_t i = 0; i < vfh_data.rows; ++i)
                    {
                        for (size_t j = 0; j < vfh_data.cols; ++j)
                        {
                            vfh_data[i][j] = vfh_models[i].second[j];
                        }
                    }
                    flann::save_to_file (vfh_data, training_vfh_data_h5_file_name, "training_vfh_data");
                    std::ofstream fs_vfh;
                    fs_vfh.open (training_vfh_data_list_file_name.c_str ());
                    for (size_t i = 0; i < vfh_models.size (); ++i)
                    {
                        fs_vfh << vfh_models[i].first << "\n";
                    }
                    fs_vfh.close ();
                    // Build the tree index and save it to disk
                    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_vfh_idx_file_name.c_str (), (int)vfh_data.rows);
                    flann::Index<flann::ChiSquareDistance<float> > vfh_index (vfh_data, flann::LinearIndexParams ());
                    //flann::Index<flann::ChiSquareDistance<float> > index (vfh_data, flann::KDTreeIndexParams (4));
                    vfh_index.buildIndex ();
                    vfh_index.save (kdtree_vfh_idx_file_name);
                    delete[] vfh_data.ptr ();

                    flann::Matrix<float> esf_data (new float[esf_models.size () * esf_models[0].second.size ()], esf_models.size (), esf_models[0].second.size ());

                    for (size_t i = 0; i < esf_data.rows; ++i)
                    {
                        for (size_t j = 0; j < esf_data.cols; ++j)
                        {
                            esf_data[i][j] = esf_models[i].second[j];
                        }
                    }
                    flann::save_to_file (esf_data, training_esf_data_h5_file_name, "training_esf_data");
                    std::ofstream fs_esf;
                    fs_esf.open (training_esf_data_list_file_name.c_str ());
                    for (size_t i = 0; i < esf_models.size (); ++i)
                    {
                        fs_esf << esf_models[i].first << "\n";
                    }
                    fs_esf.close ();
                    // Build the tree index and save it to disk
                    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_esf_idx_file_name.c_str (), (int)esf_data.rows);
                    flann::Index<flann::ChiSquareDistance<float> > esf_index (esf_data, flann::LinearIndexParams ());
                    //flann::Index<flann::ChiSquareDistance<float> > index (esf_data, flann::KDTreeIndexParams (4));
                    esf_index.buildIndex ();
                    esf_index.save (kdtree_esf_idx_file_name);
                    delete[] esf_data.ptr ();

                    flann::Matrix<float> cvfh_data (new float[cvfh_models.size () * cvfh_models[0].second.size ()], cvfh_models.size (), cvfh_models[0].second.size ());

                    for (size_t i = 0; i < cvfh_data.rows; ++i)
                    {
                        for (size_t j = 0; j < cvfh_data.cols; ++j)
                        {
                            cvfh_data[i][j] = cvfh_models[i].second[j];
                        }
                    }
                    flann::save_to_file (cvfh_data, training_cvfh_data_h5_file_name, "training_cvfh_data");
                    std::ofstream fs_cvfh;
                    fs_cvfh.open (training_cvfh_data_list_file_name.c_str ());
                    for (size_t i = 0; i < cvfh_models.size (); ++i)
                    {
                        fs_cvfh << cvfh_models[i].first << "\n";
                    }
                    fs_cvfh.close ();
                    // Build the tree index and save it to disk
                    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_cvfh_idx_file_name.c_str (), (int)cvfh_data.rows);
                    flann::Index<flann::ChiSquareDistance<float> > cvfh_index (cvfh_data, flann::LinearIndexParams ());
                    //flann::Index<flann::ChiSquareDistance<float> > index (cvfh_data, flann::KDTreeIndexParams (4));
                    cvfh_index.buildIndex ();
                    cvfh_index.save (kdtree_cvfh_idx_file_name);
                    delete[] cvfh_data.ptr ();

                }
                //std::cout <<  << std::endl;
                std::cout << vfh.first.c_str() << std::endl;
                count++;
            }
            return ecto::OK;
		}

        ecto::spore<cv::Mat> R_, T_;
        ecto::spore<ecto::pcl::FeatureCloud> features_;
        ecto::spore<ecto::pcl::PointCloud> input_;
        ecto::spore<bool> last_, novel_;

        std::vector<vfh_model> vfh_models;
        std::vector<cvfh_model> cvfh_models;
        std::vector<esf_model> esf_models;

        int count;


	};//struct Trainer
}

ECTO_CELL(ecto_msd_pcl, ecto_msd_pcl::Trainer, "Trainer", "Train the msd_pcl object detection algorithm.")
