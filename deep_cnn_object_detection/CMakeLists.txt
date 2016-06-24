cmake_minimum_required(VERSION 2.8.3)
project(deep_cnn_object_detection)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x") # Need C++11 
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  #opencv2
  cv_bridge
  tf
  tf_conversions
  message_generation
)

find_package(FLANN REQUIRED)
find_package(PCL 1.7 REQUIRED)

add_service_files(
  FILES
  getImage.srv
  UpdateTFs.srv
  SaveCloud.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

catkin_package(

)



include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(IMAGE_FILES src/image_preparator.cpp)
add_executable(image_prepare ${IMAGE_FILES})
target_link_libraries(image_prepare ${catkin_LIBRARIES}
                                           ${QT_LIBRARIES} 
                                           ${PCL_LIBRARIES} )
                                           
set(CLOUD_FILES src/cloud_saver.cpp)
add_executable(cloud_saver ${CLOUD_FILES})
target_link_libraries(cloud_saver ${catkin_LIBRARIES}
                                           ${QT_LIBRARIES} 
                                           ${PCL_LIBRARIES} )