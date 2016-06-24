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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

//#include <object_recognition_core/common/pose_result.h>
//#include <object_recognition_core/common/types.h>
//#include <object_recognition_core/common/json.hpp>
//#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>
#include <object_recognition_core/db/ModelReader.h>

#include <object_recognition_msgs/RecognizedObjectArray.h>

#include "persistence_utils.h"
#include "vtk_model_sampling.h"

#include "TrainData.h"

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::db::DocumentId;
using object_recognition_core::common::PoseResult;



namespace ecto_msd_pcl
{
    struct Detection : public object_recognition_core::db::bases::ModelReaderBase
    {
        virtual void parameter_callback(const object_recognition_core::db::Documents& db_documents)
        {
            std::cout << "loading data\n";
            std::cout << "Documents size: " <<  db_documents.size() << "\n";
            DataLoaded = false;
            BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents)
            {
                Object _Object;

                std::string object_id = document.get_field<ObjectId>("object_id");
                _Object.SetID(object_id);

                object_recognition_core::db::data readed_data;
                document.get_attachment<object_recognition_core::db::data>("data", readed_data);
                _Object.SetTrainingData(readed_data);

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud__ (new pcl::PointCloud<pcl::PointXYZ>);
                document.get_attachment<pcl::PointCloud<pcl::PointXYZ> >("model_cloud", *cloud__);
                _Object.SetCloud(cloud__);

                _Detector.AddObject(_Object);

                DataLoaded = true;
                std::cout << "Object loaded\n";
            }
            if (DataLoaded)
            {
                _Detector.BuildFlannIndex();
                std::cout << "Index is built\n";
            }
            std::cout << "Parameter_callback end\n";
        }
        static void declare_params(ecto::tendrils& params)
        {
            object_recognition_core::db::bases::declare_params_impl(params, "msd_pcl");
        }
        static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
        {
            inputs.declare<ecto::pcl::PointCloud>("input", "Input point cloud").required(true);
            outputs.declare(&Detection::pose_results_, "pose_results", "The results of object recognition");
        }
        void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
        {
            configure_impl();
            input_ = inputs["input"];
        }
        int process(const tendrils& inputs, const tendrils& outputs)
        {
            std::cout << "Process start\n";
	    pose_results_->clear();
            int ObjectsNumber = _Detector.GetObjectsNumber();

            if (ObjectsNumber == 0)
            {
                return ecto::QUIT;
            }
            ecto::pcl::xyz_cloud_variant_t cv = input_->make_variant();
            __input_cloud = boost::get< boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > >(cv);

            PointCloudProcessor _PointCloudProcessor;
            std::cout << "Processing input cloud \n";
            _PointCloudProcessor.SetInputCloud(__input_cloud);
            std::vector<PointCloudCluster> _Clusters = _PointCloudProcessor.GetClusters();
            std::cout << "Start looking for objects \n";
            _Detector.SetDB(db_);
            _Detector.setClusters(_Clusters);
            _Detector.DetectObject();
            if (_Detector.GetPoseReuslts().size() > 0)
	    {
                *pose_results_ = _Detector.GetPoseReuslts();
	    }
            return 0;
        }

        boost::shared_ptr<const pcl::PointCloud< pcl::PointXYZRGB> > __input_cloud;
        ecto::spore<ecto::pcl::PointCloud> input_;
        ecto::spore<std::vector<object_recognition_core::common::PoseResult> > pose_results_;
        object_recognition_core::db::data data_;
        Detector _Detector;

        bool DataLoaded;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };//struct Detection
}

ECTO_CELL(ecto_msd_pcl, ecto_msd_pcl::Detection, "Detector", "Detect the object by msd_pcl detection algorithm.")



