#include <ecto/ecto.hpp>
#include <iostream>
#include <ecto/ecto.hpp>
#include <ecto_pcl/ecto_pcl.hpp>
#include <fstream>
#include <boost/format.hpp>
#include <boost/format/free_funcs.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

namespace ecto
{
	namespace pcl
	{
		template<typename PointT> inline void writePLY(const ::pcl::PointCloud<PointT>& cloud_m, const std::string& mesh_file_name)
		{
			
			::pcl::io::savePLYFileBinary(std::string(mesh_file_name).c_str(), cloud_m);
			/*std::cout << "1";
			std::ofstream mesh_file(std::string(mesh_file_name).c_str());
			mesh_file << "ply\n"
			"format ascii 1.0\n"
			"element vertex "
			<< cloud_m.points.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"end_header\n";

			//<x> <y> <z> <r> <g> <b>
			for (size_t i = 0; i < cloud_m.points.size(); i++)
			{
				const PointT& p = cloud_m.points[i];
				mesh_file << p.x << " " << p.y << " " << p.z << "\n";
			}*/
		}
		
		template<> inline void writePLY<CloudPOINTXYZRGB::PointType>(const CloudPOINTXYZRGB& cloud_m, const std::string& mesh_file_name)
		{
			::pcl::io::savePLYFileBinary(std::string(mesh_file_name).c_str(), cloud_m);
			/*std::ofstream mesh_file(std::string(mesh_file_name).c_str());
			mesh_file << "ply\n"
			"format ascii 1.0\n"
			"element vertex "
			<< cloud_m.points.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property uchar red\n"
			"property uchar green\n"
			"property uchar blue\n"
			"end_header\n";
			//<x> <y> <z> <r> <g> <b>
			for (size_t i = 0; i < cloud_m.points.size(); i++)
			{
				const CloudPOINTXYZRGB::PointType& p = cloud_m.points[i];
				mesh_file << p.x << " " << p.y << " " << p.z << " " << int(p.r) << " " << int(p.g) << " " << int(p.b) << "\n";
			}*/
		}

		template<> inline void writePLY<CloudPOINTNORMAL::PointType>(const CloudPOINTNORMAL& cloud_m, const std::string& mesh_file_name)
		{
			::pcl::io::savePLYFileBinary(std::string(mesh_file_name).c_str(), cloud_m);
			/*std::cout << "3";
			std::ofstream mesh_file(std::string(mesh_file_name).c_str());
			mesh_file << "ply\n"
			"format ascii 1.0\n"
			"element vertex "
			<< cloud_m.points.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property float nx\n"
			"property float ny\n"
			"property float nz\n"
			"end_header\n";

			//<x> <y> <z> <r> <g> <b>
			for (size_t i = 0; i < cloud_m.points.size(); i++)
			{
				const CloudPOINTNORMAL::PointType& p = cloud_m.points[i];
				mesh_file << p.x << " " << p.y << " " << p.z << " "
				 << p.normal_x << " " << p.normal_y << " " << p.normal_z << "\n";
			}*/
		}

		template<> inline void writePLY<CloudPOINTXYZRGBNORMAL::PointType>(const CloudPOINTXYZRGBNORMAL& cloud_m, const std::string& mesh_file_name)
		{
			::pcl::io::savePLYFileBinary(std::string(mesh_file_name).c_str(), cloud_m);
			/*std::cout << "4";
			std::ofstream mesh_file(std::string(mesh_file_name).c_str());
			mesh_file << "ply\n"
			"format ascii 1.0\n"
			"element vertex "
			<< cloud_m.points.size() << "\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property uchar red\n"
			"property uchar green\n"
			"property uchar blue\n"
			"property float nx\n"
			"property float ny\n"
			"property float nz\n"
			"end_header\n";

			//<x> <y> <z> <r> <g> <b>
			for (size_t i = 0; i < cloud_m.points.size(); i++)
			{
				const CloudPOINTXYZRGBNORMAL::PointType& p = cloud_m.points[i];
				mesh_file << p.x << " " << p.y << " " << p.z << " "
				<< int(p.r) << " " << int(p.g) << " " << int(p.b) << " "
				<< p.normal_x << " " << p.normal_y << " " << p.normal_z << "\n";
			}*/
		}

		struct PLYWriter_Bin
		{
			PLYWriter_Bin():count_(0)
			{
			}

			static void	declare_params(tendrils& params)
			{
				params.declare<std::string>("filename_format", "The format string for saving PLY files, "
								"must succeed with a single unsigned int argument.",
								"cloud_%04u.ply");
			}

			static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
			{
				inputs.declare<PointCloud>("input", "A point cloud to put in a pcd file.");
			}

			void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
			{
				input_ = inputs["input"];
				filename_format_ = params["filename_format"];
			}

			struct write_dispatch: boost::static_visitor<>
			{
				std::string file;
				write_dispatch(std::string f):file(f)
				{
				}
				template<typename CloudType> void operator()(CloudType& cloud) const
				{
					writePLY(*cloud, file);
				}
			};

			int	process(const tendrils& /*inputs*/, const tendrils& /*outputs*/)
			{
				std::string filename = boost::str(boost::format(*filename_format_) % count_++);
				xyz_cloud_variant_t cv = input_->make_variant();
				boost::apply_visitor(write_dispatch(filename), cv);
				return ecto::OK;
			}
			spore<PointCloud> input_;
			spore<std::string> filename_format_;
			unsigned count_;
		};
	}
}
// This macro is required to register the cell with the module
// first argument: the module that was defined in the tutorial.cpp
// second argument: the cell we want to expose in the module
// third argument: the name of that cell as seen in Python
// fourht argument: a description of what that cell does
ECTO_CELL(PLYWriter_Bin, ecto::pcl::PLYWriter_Bin, "PLYWriter_Bin", "Prints 'Hello' to standard output.");
