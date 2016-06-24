cmake_minimum_required(VERSION 2.8.3)
project(ork_to_pose_converter)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x") # Need C++11 


find_package(catkin REQUIRED COMPONENTS  
  roscpp
  rospy
  std_msgs  
  visualization_msgs
  object_recognition_msgs
  object_recognition_ros
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES operation_panel
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
find_package(catkin REQUIRED)
find_package(cmake_modules REQUIRED)



include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN_INCLUDE_DIRS} 
)



add_executable(ork_to_pose_converter_node src/ork_to_pose_converter_node.cpp)
target_link_libraries(ork_to_pose_converter_node 
	    ${catkin_LIBRARIES}
            )


