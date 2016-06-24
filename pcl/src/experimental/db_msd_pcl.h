
#ifndef DB_MSD_PCL_H_
#define DB_MSD_PCL_H_

#include <object_recognition_core/db/document.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
typedef std::pair<std::string, std::vector<float> > vfh_model;

namespace object_recognition_core
{
    namespace db
    {
        struct data
        {
            std::vector<Eigen::Vector3f> centroids;
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > roll_transforms;
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
            std::vector<std::string> ids;
            std::vector<std::vector<float> > histograms;
        };
    // Specializations for pcl::PointCloud<pcl::PointXYZRGB>

        template<> void object_recognition_core::db::DummyDocument::set_attachment<data >(const AttachmentName& attachment_name,
                                                                                          const data& value);
        template<> void object_recognition_core::db::DummyDocument::set_attachment<pcl::PointCloud<pcl::PointXYZ> >(const AttachmentName& attachment_name,
                                                                                          const pcl::PointCloud<pcl::PointXYZ>& value);
        template<> void object_recognition_core::db::DummyDocument::get_attachment<data >(const AttachmentName& attachment_name,
                                                                                             data& value) const;
        template<> void object_recognition_core::db::DummyDocument::get_attachment<pcl::PointCloud<pcl::PointXYZ> >(const AttachmentName& attachment_name,
                                                                                             pcl::PointCloud<pcl::PointXYZ> & value) const;
    }
}

#endif /* DB_MSD_PCL_H_ */
