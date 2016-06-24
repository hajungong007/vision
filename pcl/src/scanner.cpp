#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

#include <ecto/ecto.hpp>

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

namespace ecto_msd_pcl
{
	struct Scanner
	{
		static void	declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
		{

		}

		void configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
		{

		}

		int process(const tendrils& inputs, const tendrils& outputs)
		{
			return 1;
		}

	};
} 

ECTO_CELL(ecto_msd_pcl, ecto_msd_pcl::Scanner, "Trainer", "Scan object for msd_pcl algorithm.")
