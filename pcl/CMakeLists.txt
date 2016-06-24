cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_msd_pcl)

find_package(catkin)
find_package(catkin REQUIRED ecto object_recognition_core)
find_package(catkin REQUIRED COMPONENTS)
#find_package(PCL REQUIRED)

catkin_package(CATKIN_DEPENDS ecto object_recognition_core)

add_definitions("-Wall -Werror -Wl,--no-undefined ")

find_package(FLANN REQUIRED)

find_package(ecto REQUIRED)
find_package(ecto_opencv REQUIRED)
find_package(object_recognition_core REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(HDF5 REQUIRED)


include_directories(${catkin_INCLUDE_DIRS} include)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


include_directories(${FLANN_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
catkin_python_setup()

add_subdirectory(python)
add_subdirectory(src)
add_subdirectory(PLY_WRITER_BIN)

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# install the applications
install(PROGRAMS ${PROJECT_SOURCE_DIR}/apps/object_scanner
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# build the docs
ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/source)

# install the conf files
install(DIRECTORY conf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
