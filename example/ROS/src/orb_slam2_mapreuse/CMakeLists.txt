cmake_minimum_required(VERSION 3.0.2)
project(orb_slam2_mapreuse)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  tf
)

find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  "/home/antusheng/research/map_based_localization/test/ORB_SLAM2_MapReuse"
  "/home/antusheng/research/map_based_localization/test/ORB_SLAM2_MapReuse/include"
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Boost_LIBRARIES}
${Pangolin_LIBRARIES}
/home/antusheng/research/map_based_localization/test/ORB_SLAM2_MapReuse/Thirdparty/DBoW2/lib/libDBoW2.so
/home/antusheng/research/map_based_localization/test/ORB_SLAM2_MapReuse/Thirdparty/g2o/lib/libg2o.so
/home/antusheng/research/map_based_localization/test/ORB_SLAM2_MapReuse/lib/libORB_SLAM2_MapReuse.so
)

add_executable(orb_mapping_node src/orb_mapping_node.cpp)
target_link_libraries(orb_mapping_node
  ${catkin_LIBRARIES}
  ${LIBS}
)

add_executable(orb_slam_node src/orb_slam_node.cpp)
target_link_libraries(orb_slam_node
  ${catkin_LIBRARIES}
  ${LIBS}
)

# add_executable(orb_localization_node src/orb_localization_node.cpp)
# target_link_libraries(orb_localization_node
#   ${catkin_LIBRARIES}
#   ${LIBS}
# )
