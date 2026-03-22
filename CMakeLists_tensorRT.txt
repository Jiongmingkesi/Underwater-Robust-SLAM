cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -fopenmp -march=native -pthread")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS} -g")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV 4.4 REQUIRED COMPONENTS core imgproc highgui features2d calib3d)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(Gflags REQUIRED)

# Glog settings
find_path(GLOG_INCLUDE_DIR NAMES glog/logging.h HINTS /usr/local/include /usr/include)
find_library(GLOG_LIBRARY NAMES glog HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
if (NOT GLOG_INCLUDE_DIR OR NOT GLOG_LIBRARY)
    message(FATAL_ERROR "Google Logging (glog) not found.")
endif ()
set(Glog_FOUND TRUE)
set(Glog_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
set(Glog_LIBRARIES ${GLOG_LIBRARY})

# ==================== [修改 1] 添加 TensorRT 查找逻辑 ====================
# Jetson 通常在 /usr/include/aarch64-linux-gnu 或 /usr/include 中
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/cuda/include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/cuda/lib64)

# 某些情况下还需要 nvparsers (如果你用了onnx parser) 或 nvinfer_plugin
# 但鉴于你直接加载 .engine 文件，通常只需要 nvinfer 即可
# 如果运行报错找不到其他符号，可以解开下面的注释：
# find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/lib)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "Found TensorRT header: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "Found TensorRT lib: ${TENSORRT_LIBRARY}")
else()
    message(FATAL_ERROR "TensorRT not found! Please check path.")
endif()
# =======================================================================

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}

$ENV{ONNXRUNTIME_ROOT}/include
${TENSORRT_INCLUDE_DIR} 
${CUDA_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${GFLAGS_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED 
src/Random.cpp
src/Timestamp.cpp
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/Extractors/SPextractor.cc
src/Extractors/superpoint_onnx.cc
src/Matchers/SPmatcher.cc
src/Matchers/lightglue_onnx.cpp
src/Matchers/transform.cpp
include/Timestamp.h
include/Random.h
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Extractors/SPextractor.h
include/Extractors/superpoint_onnx.h
include/Matchers/SPmatcher.h
include/Matchers/lightglue_onnx.h
include/Matchers/Configuration.h
include/Matchers/transform.h
include/Settings.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${CUDA_LIBRARIES}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3/lib/libDBoW3.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto

$ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
${TENSORRT_LIBRARY}
# 如果编译报错 undefined reference，可能需要解开下面的 plugin 库
# ${TENSORRT_PLUGIN_LIBRARY}
)

# Build examples
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-return-type -Wno-unused-parameter")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})