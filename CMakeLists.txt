cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

## 能怎样？  修改
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")  ## 解决编译时的 未初始化警告问题
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") ## 强制删除 Werror



set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3") #其中${CMAKE_C_FLAGS}表示先前在CMake中定义的C语言编译器标志，而-Wall -O3是额外添加到这个变量中的编译器标志。
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -fopenmp -march=native -pthread")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS} -g")
# Check C++11 or C++0x support

# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#    add_definitions(-DCOMPILEDWITHC11)
#    message(STATUS "Using flag -std=c++11.")
# elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#    add_definitions(-DCOMPILEDWITHC0X)
#    message(STATUS "Using flag -std=c++0x.")
# else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# 要保证整个工程的opencv版本一致，包括dbow，源码以及ros相关的
# 3 4 都可以正常运行//必须要opencv3.4.13,要不然会在加载词袋处卡住
## find_package(OpenCV 3.4 REQUIRED)
# find_package(OpenCV 3.4 REQUIRED)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 3.2 not found.")
#    endif()

set(OpenCV_DIR "/home/jiongkesi2/SLAM_Ev/opencv-4.8.1/lib/lib/cmake/opencv4")


find_package(OpenCV 4.4 REQUIRED COMPONENTS core imgproc highgui features2d calib3d)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()





# 打印 OpenCV 版本和路径信息
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "Found OpenCV install path: ${OpenCV_INSTALL_PATH}")
message(STATUS "Found OpenCV config file in: ${OpenCV_DIR}")



MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})


find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(Gflags REQUIRED)
# find_package(Glog REQUIRED)

# find_package(Glog REQUIRED)  ## 找不到 自己设置了
# Manually find Google Logging (glog)
##############  以下是设置 Glog 的 include 和 lib 路径
find_path(GLOG_INCLUDE_DIR NAMES glog/logging.h
    HINTS /usr/local/include /usr/include
)

find_library(GLOG_LIBRARY NAMES glog
    HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

if (NOT GLOG_INCLUDE_DIR OR NOT GLOG_LIBRARY)
    message(FATAL_ERROR "Google Logging (glog) not found.")
endif ()

set(Glog_FOUND TRUE)
set(Glog_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
set(Glog_LIBRARIES ${GLOG_LIBRARY})
##############  以上是设置 Glog 的 include 和 lib 路径



# find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
$ENV{ONNXRUNTIME_ROOT}/include
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
## src/Extractors/export_voc.cc   ## 输出词典训练数据
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
## include/Extractors/export_voc.h   ## 输出词典训练数据
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
# /usr/local/lib/libonnxruntime.so
$ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)


# Build examples


# Build examples
## 能怎样？  修改
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")  ## 解决编译时的 未初始化警告问题
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-return-type -Wno-unused-parameter")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") ## 强制删除 Werror




#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})


#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})


add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})