# Compile the ecto
find_package(PCL 1.7 REQUIRED)
find_package(HDF5 REQUIRED)
find_package(FLANN REQUIRED)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
			               ${FLANN_INCLUDE_DIRS}
                           ${HDF5_INCLUDE_DIR}
                       #/usr/local/hdf5/include                           
)

add_library(PCLProcessor stable/PCLProcessor.cpp)
link_libraries(PCLProcessor ${PCL_LIBRARIES})


add_library(TrainData stable/TrainData.cpp)
link_libraries(TrainData ${PCL_LIBRARIES} ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

ectomodule(ecto_msd_pcl DESTINATION ${PROJECT_NAME}/ecto_cells
                        INSTALL
           stable/detector.cpp
           stable/trainer.cpp
           stable/module.cpp
           experimental/db_msd_pcl.cpp
)

link_ecto(ecto_msd_pcl ${catkin_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        ${PCL_LIBRARIES} 
                        ${Boost_LIBRARIES} 
                        ${FLANN_LIBRARIES} 
                        ${HDF5_hdf5_LIBRARY}
                        PCLProcessor
                        TrainData
)

ectomodule(ecto_msd_pcl_exp DESTINATION ${PROJECT_NAME}/ecto_cells
                        INSTALL
            experimental/detector_experimental.cpp
            experimental/vfh_trainer_experimental.cpp
            experimental/module_experimental.cpp
            experimental/db_msd_pcl.cpp
            experimental/CAD_trainer.cpp            
)

link_ecto(ecto_msd_pcl_exp ${catkin_LIBRARIES}
                            ${OpenCV_LIBRARIES}
                            ${PCL_LIBRARIES} 
                            ${Boost_LIBRARIES} 
                            ${FLANN_LIBRARIES} 
                            ${HDF5_hdf5_LIBRARY}
)

