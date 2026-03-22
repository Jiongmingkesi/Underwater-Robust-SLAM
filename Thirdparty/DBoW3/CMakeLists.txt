cmake_minimum_required(VERSION 2.8)
project(DBoW3)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall  -O3 -march=native")

set(HDRS_DBOW3
  src/BowVector.h
  src/Database.h
  src/DBoW3.h
  src/DescManip.h
  src/exports.h
  src/QueryResults.h
  src/quicklz.h
  src/timers.h
  src/FeatureVector.h
  src/ScoringObject.h   
  src/Vocabulary.h)
set(SRCS_DBOW3
  src/BowVector.cpp
  src/Database.cpp
  src/DescManip.cpp
  src/QueryResults.cpp
  src/quicklz.c
  src/FeatureVector.cpp
  src/ScoringObject.cpp   
  src/Vocabulary.cpp)


# find_package(OpenCV 4 REQUIRED)
set(OpenCV_DIR "/home/jiongkesi2/SLAM_Ev/opencv-4.8.1/lib/lib/cmake/opencv4")

find_package(OpenCV 4.4 REQUIRED COMPONENTS core imgproc features2d)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW3 SHARED ${SRCS_DBOW3})
target_link_libraries(DBoW3 ${OpenCV_LIBS})



# Detect optional xfeatures2d (opencv_contrib)
find_package(OpenCV COMPONENTS xfeatures2d QUIET)
if(OpenCV_FOUND AND OpenCV_LIBS MATCHES "xfeatures2d")
  add_definitions(-DHAVE_OPENCV_XFEATURES2D)
  MESSAGE(STATUS "DBoW3: OpenCV xfeatures2d found.")
else()
  MESSAGE(STATUS "DBoW3: OpenCV xfeatures2d NOT found. SURF tests will fallback to ORB.")
endif()
