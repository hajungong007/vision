
#include <boost/filesystem.hpp>

#include <object_recognition_core/db/opencv.h>

#include "db_msd_pcl.h"

#include <pcl/io/pcd_io.h>
typedef std::pair<std::string, std::vector<float> > vfh_model;
namespace
{
  object_recognition_core::db::MimeType MIME_TYPE = "text/x-yaml";
}
namespace object_recognition_core
{
    namespace db
    {
        template<> void object_recognition_core::db::DummyDocument::set_attachment<data >(const AttachmentName& attachment_name,
                                                                                          const data& value)
        {

            std::stringstream out;

            out << "centroids: " << value.centroids.size() << std::endl;
            out << "roll_transforms: " << value.roll_transforms.size() << std::endl;
            out << "ids: " << value.ids.size() << std::endl;
            out << "poses: " << value.poses.size() << std::endl;
            out << "histograms: " << value.histograms.size() << std::endl;

            for (size_t i = 0; i < value.centroids.size(); i++)
            {
                out << "centroid_" << value.ids[i] << ": " <<  value.centroids[i][0] << " " << value.centroids[i][1] << " " << value.centroids[i][2] << "\n";
            }
            for (size_t i = 0; i < value.roll_transforms.size(); i++)
            {
                out << "roll_transform_" << value.ids[i] << ": "
                    << value.roll_transforms[i](0,0) << " " << value.roll_transforms[i](0,1) << " " << value.roll_transforms[i](0,2) << " " << value.roll_transforms[i](0,3) << " "
                    << value.roll_transforms[i](1,0) << " " << value.roll_transforms[i](1,1) << " " << value.roll_transforms[i](1,2) << " " << value.roll_transforms[i](1,3) << " "
                    << value.roll_transforms[i](2,0) << " " << value.roll_transforms[i](2,1) << " " << value.roll_transforms[i](2,2) << " " << value.roll_transforms[i](2,3) << " "
                    << value.roll_transforms[i](3,0) << " " << value.roll_transforms[i](3,1) << " " << value.roll_transforms[i](3,2) << " " << value.roll_transforms[i](3,3) << "\n";
            }
            for (size_t i = 0; i < value.poses.size(); i++)
            {
                out << "pose_" << i << ": "
                    << value.poses[i](0,0) << " " << value.poses[i](0,1) << " " << value.poses[i](0,2) << " " << value.poses[i](0,3) << " "
                    << value.poses[i](1,0) << " " << value.poses[i](1,1) << " " << value.poses[i](1,2) << " " << value.poses[i](1,3) << " "
                    << value.poses[i](2,0) << " " << value.poses[i](2,1) << " " << value.poses[i](2,2) << " " << value.poses[i](2,3) << " "
                    << value.poses[i](3,0) << " " << value.poses[i](3,1) << " " << value.poses[i](3,2) << " " << value.poses[i](3,3) << "\n";
            }
            for (size_t i = 0; i < value.histograms.size(); i++)
            {
                out << "histogram_" << value.ids[i] << ": ";
                for (size_t j = 0; j < value.histograms[i].size(); j++)
                {
                    out << value.histograms[i][j] << " ";
                }
                out << "\n";
            }

            set_attachment_stream(attachment_name, out, MIME_TYPE);
        }
        template<> void object_recognition_core::db::DummyDocument::set_attachment<pcl::PointCloud<pcl::PointXYZ> >(const AttachmentName& attachment_name,
                                                                                      const pcl::PointCloud<pcl::PointXYZ>& value)
        {
            pcl::io::savePCDFileASCII(attachment_name, value);
            std::ifstream ifs(attachment_name.c_str(), std::ios_base::in);
            set_attachment_stream(attachment_name, ifs, MIME_TYPE);
        }

        template<> void object_recognition_core::db::DummyDocument::get_attachment<data>(const AttachmentName& attachment_name,
                                                                                          data& value) const
        {
 //           std::string file_name = temporary_yml_file_name(true);
            std::stringstream ss;
            this->get_attachment_stream(attachment_name, ss, MIME_TYPE);
            std::string data_str = ss.str();
            int ids_size = atoi(data_str.substr(data_str.find("histograms:") + 11, data_str.find("histograms:") + 15).c_str());
            int poses_size = atoi(data_str.substr(data_str.find("poses:") + 6, data_str.find("poses:") + 9).c_str());
            //poses: 63
            //histograms: 125
//            value.poses.resize(poses_size);
//            value.centroids.resize(ids_size);
//            value.roll_transforms.resize(ids_size);
//            value.ids.resize(ids_size);
//            value.histograms.resize(ids_size);

            data_str = data_str.substr(data_str.find("centroid_"));
//Read centroids and ids
            for (int i = 0; i < ids_size; i++)
            {
                std::size_t id_end = data_str.find(":");
                value.ids.push_back(data_str.substr(9, id_end - 9));

                std::size_t first_space = data_str.find_first_of(" ", id_end);
                std::size_t second_space = data_str.find_first_of(" ", first_space + 1);
                std::size_t third_space = data_str.find_first_of(" ", second_space + 1);
                std::size_t end_string = data_str.find_first_of("\n", third_space + 1);

                float centr1 = atof( data_str.substr(first_space + 1, second_space).c_str());
                float centr2 = atof( data_str.substr(second_space + 1, third_space).c_str());
                float centr3 = atof( data_str.substr(third_space + 1, end_string).c_str());

                Eigen::Vector3f centroid_temp (centr1, centr2, centr3);
                value.centroids.push_back(centroid_temp);
                if (i < ids_size - 1)

                {
                    std::size_t new_data = data_str.find("centroid_", id_end);
                    data_str = data_str.substr(new_data);
                }
                else if ( i == ids_size -1)
                {
                    std::size_t new_data = data_str.find("roll_transform_0_0:");
                    data_str = data_str.substr(new_data);
                }
            }
//Read roll transforms
            for (int i = 0; i < ids_size; i++)
            {
                Eigen::Matrix4f roll_transform;
                for (int j = 0; j < 15; j++)
                {
                    std::size_t first_space = data_str.find_first_of(" ");
                    std::size_t second_space = data_str.find_first_of(" ", first_space + 1);
                    roll_transform ( j / 4, j % 4) = atof( data_str.substr( first_space + 1, second_space).c_str());
                    data_str = data_str.substr(second_space);
                }
                std::size_t first_space = data_str.find_first_of(" ");
                std::size_t end_string = data_str.find_first_of("\n", first_space + 1);
                roll_transform ( 3, 3) = atof( data_str.substr( first_space + 1, end_string).c_str());
                data_str = data_str.substr(end_string);
                value.roll_transforms.push_back(roll_transform);
            }
//Read poses
            for (int i = 0; i < poses_size; i++)
            {
                Eigen::Matrix4f pose;
                for (int j = 0; j < 15; j++)
                {
                    std::size_t first_space = data_str.find_first_of(" ");
                    std::size_t second_space = data_str.find_first_of(" ", first_space + 1);
                    pose ( j / 4, j % 4) = atof( data_str.substr( first_space + 1, second_space).c_str());
                    data_str = data_str.substr(second_space);
                }
                std::size_t first_space = data_str.find_first_of(" ");
                std::size_t end_string = data_str.find_first_of("\n", first_space + 1);
                pose ( 3, 3) = atof( data_str.substr( first_space + 1, end_string).c_str());
                data_str = data_str.substr(end_string);
                value.poses.push_back(pose);
            }
//Read histograms
            for (int i = 0; i < ids_size; i++)
            {
               std::vector<float> histogram;
               histogram.resize(308);
                for (int j = 0; j < 307; j++)
                {
                    std::size_t first_space = data_str.find_first_of(" ");
                    std::size_t second_space = data_str.find_first_of(" ", first_space + 1);
                    histogram[j] = atof( data_str.substr( first_space + 1, second_space).c_str());
                    data_str = data_str.substr(second_space);
                }
                std::size_t first_space = data_str.find_first_of(" ");
                std::size_t end_string = data_str.find_first_of("\n", first_space + 1);
                histogram[307] = atof( data_str.substr( first_space + 1, end_string).c_str());
                data_str = data_str.substr(end_string);
                value.histograms.push_back(histogram);
            }

//            std::cout << "Histograms: " << value.histograms.size() << "\n";
//            std::cout << "roll_transform: " << value.roll_transforms.size() << "\n";
//            std::cout << "poses: " << value.poses.size() << "\n";
//            std::cout << "centroids: " << value.centroids.size() << "\n";
//            std::cout << "ids: " << value.ids.size() << "\n";
        }
        template<> void object_recognition_core::db::DummyDocument::get_attachment<pcl::PointCloud<pcl::PointXYZ> >(const AttachmentName& attachment_name,
                                                                                          pcl::PointCloud<pcl::PointXYZ> & value) const
        {

            std::string file_name = temporary_yml_file_name(true);
            std::stringstream ss;
            this->get_attachment_stream(attachment_name, ss, MIME_TYPE);

            // Write it to disk
            std::ofstream writer(file_name.c_str());
            writer << ss.rdbuf() << std::flush;

            pcl::io::loadPCDFile(file_name.c_str(), value);
        }
    }
}
