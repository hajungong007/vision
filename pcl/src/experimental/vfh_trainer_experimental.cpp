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

#include "db_msd_pcl.h"

using ecto::tendrils;
using ecto::spore;
typedef std::pair<std::string, std::vector<float> > vfh_model;
using object_recognition_core::db::Document;
using object_recognition_core::db::DocumentId;

namespace ecto_msd_pcl
{
    struct VFH_Trainer_experimental
    {
        VFH_Trainer_experimental()
        {
            count = 0;
            iter_count = 0;
            novel_received = false;
            end_please = false;
        }
        static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
        {
            inputs.declare<cv::Mat>("R", "The orientation.").required(true);
            inputs.declare<cv::Mat>("T", "The translation.").required(true);
            inputs.declare<ecto::pcl::FeatureCloud>("features", "View point features");
            inputs.declare<ecto::pcl::PointCloud>("input", "Input point cloud");
            inputs.declare<bool>("last", "True if this is a last frame");
            inputs.declare<bool>("novel", "True if frame is new");
            outputs.declare(&VFH_Trainer_experimental::json_db_out, "json_db", "Database parameters");
            outputs.declare(&VFH_Trainer_experimental::object_id_out, "object_id", "The object id, to associate this model with.");
            outputs.declare(&VFH_Trainer_experimental::db_document_, "db_document", "The filled document.");
            outputs.declare(&VFH_Trainer_experimental::commit_, "commit", "Upload to db.");

        }
        static void declare_params(ecto::tendrils& params)
        {
            params.declare(&VFH_Trainer_experimental::json_db_in, "json_db_in", "The DB parameters", "{}").required(true);
            params.declare(&VFH_Trainer_experimental::object_id_in, "object_id_in", "The object id, to associate this model with.", "{}").required(true);
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
            //Document db_document;

            if (end_please)
                return ecto::QUIT;

            *json_db_out = *json_db_in;
            *object_id_out = *object_id_in;
            *commit_ = false;
            if (*novel_ || (novel_received && iter_count < 5))
            {
                novel_received = true;

                std::string kdtree_vfh_idx_file_name = "kdtree_vfh.idx";
                std::string training_vfh_data_h5_file_name = "training_vfh_data.h5";
                std::string training_vfh_data_list_file_name = "training_vfh_data.list";

                pcl::PointCloud<pcl::PointXYZRGB> input_cloud;
                pcl::PointCloud<pcl::VFHSignature308> features;

                // convert ecto::tendrils to pcl::clouds
                ecto::pcl::xyz_cloud_variant_t cv = input_->make_variant();
                boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > __input_cloud = boost::get< boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > >(cv);
                input_cloud = *__input_cloud; //xyzrgb pointcloud

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

                std::stringstream vfh_ss;
                vfh_ss << count << "." << iter_count << ".vfh_signature.pcd";
                std::stringstream cloud_ss;
                cloud_ss << count << "." << iter_count << ".cloud.pcd";
                //pcl::io::savePCDFileASCII(cloud_ss.str(), input_cloud);
                //std::cout << "Saved pointcloud: " << cloud_ss.str() << std::endl;

                //db_document.set_attachment<pcl::PointCloud<pcl::PointXYZRGB> > (cloud_ss.str(), input_cloud); // testing pcl db

                //*db_document_ = db_document;
                //*commit_ = true;

                vfh_model vfh;

                vfh.second.resize(308);
                vfh.first = vfh_ss.str();
                for (int i = 0; i < 308; i++)
                {
                    vfh.second[i] = vfh_signature->points[0].histogram[i];
                }
                vfh_models.push_back(vfh);

                if (*last_ && iter_count == 4)
                {
                    /*flann::Matrix<float> vfh_data (new float[vfh_models.size () * vfh_models[0].second.size ()], vfh_models.size (), vfh_models[0].second.size ());

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
                    fs_vfh.close ();*/
                    // Build the tree index and save it to disk
                    //pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_vfh_idx_file_name.c_str (), (int)vfh_data.rows);
                    //flann::Index<flann::ChiSquareDistance<float> > vfh_index (vfh_data, flann::LinearIndexParams ());
                    //flann::Index<flann::ChiSquareDistance<float> > index (vfh_data, flann::KDTreeIndexParams (4));
                    //vfh_index.buildIndex ();
                    //vfh_index.save (kdtree_vfh_idx_file_name);

//                    db_document.set_attachment<std::vector<vfh_model> > ("vfh_models", vfh_models); // histograms
//                    *db_document_ = db_document;
//                    *commit_ = true;
//                    end_please = true;

                    //delete[] vfh_data.ptr ();
                }
                if (iter_count == 4)
                {
                    std::cout << "End of cycle." << std::endl;
                    iter_count = 0;
                    count++;
                    novel_received = false;
                    return ecto::OK;
                }
                iter_count++;
            }
            return ecto::OK;
        }

        ecto::spore<cv::Mat> R_, T_;
        ecto::spore<ecto::pcl::FeatureCloud> features_;
        ecto::spore<ecto::pcl::PointCloud> input_;
        ecto::spore<bool> last_, novel_, commit_;

        std::vector<vfh_model> vfh_models;

        int count;
        int iter_count;
        bool novel_received, end_please;

        object_recognition_core::db::ObjectDbPtr db;
        ecto::spore<std::string> json_db_out;
        ecto::spore<std::string> json_db_in;
        ecto::spore<Document> db_document_;
        ecto::spore<DocumentId> object_id_out;
        ecto::spore<DocumentId> object_id_in;

        Document db_document;

    };//struct Trainer
}

ECTO_CELL(ecto_msd_pcl_exp, ecto_msd_pcl::VFH_Trainer_experimental, "VFH_Trainer_experimental", "Trainer_experimental the msd_pcl object detection algorithm.")
