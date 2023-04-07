cmake_minimum_required(VERSION 2.8.0)
project(ORB_SLAM2_MapReuse)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(Boost REQUIRED COMPONENTS serialization)
if(NOT Boost_FOUND)
   message("Boost not found.")
endif()

find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch/share/cmake/Torch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/Locator.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/ORBrefiner.cc
)

target_link_libraries(${PROJECT_NAME} PUBLIC
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${Boost_LIBRARIES}
${TORCH_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

# Build examples


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example/CPP)

add_executable(orb_slam_mono example/CPP/orb_slam_mono.cc)
target_link_libraries(orb_slam_mono ${PROJECT_NAME})

add_executable(orb_mapping_mono example/CPP/orb_mapping_mono.cc)
target_link_libraries(orb_mapping_mono ${PROJECT_NAME})

add_executable(orb_localization_mono example/CPP/orb_localization_mono.cc)
target_link_libraries(orb_localization_mono ${PROJECT_NAME})

add_executable(mono_euroc example/CPP/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(stereo_euroc example/CPP/stereo_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_euroc_localization example/CPP/stereo_euroc_localization.cpp)
target_link_libraries(stereo_euroc_localization ${PROJECT_NAME})

add_executable(mono_kitti example/CPP/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(stereo_kitti example/CPP/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(test_libtorch example/CPP/test_libtorch.cpp src/ORBrefiner.cc)
target_link_libraries(test_libtorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(train_bow example/CPP/train_bow.cpp src/ORBextractor.cc src/ORBrefiner.cc)
target_link_libraries(train_bow ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)

add_executable(test_sp example/CPP/test_sp.cpp)
target_link_libraries(test_sp ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_orb example/CPP/test_orb.cpp)
target_link_libraries(test_orb ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_sp_extract example/CPP/test_sp_extract.cpp src/SuperPoint.cc)
target_link_libraries(test_sp_extract ${TORCH_LIBRARIES} ${OpenCV_LIBS})