echo "Building ROS nodes"
# 设置环境变量，让 cmake 优先使用你的 workspace
export ROS_PACKAGE_PATH=:/home/jiongkesi2/work_space2/catkin_ws/src/vision_opencv/cv_bridge/build/src:$ROS_PACKAGE_PATH
export LD_LIBRARY_PATH=/home/jiongkesi2/work_space2/catkin_ws/src/vision_opencv/cv_bridge/build/devel/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/home/jiongkesi2/work_space2/catkin_ws/src/vision_opencv/cv_bridge/build/devel:/opt/ros/noetic

# export LD_LIBRARY_PATH=/home/jiongkesi2/work_space2/catkin_ws/src/vision_opencv/cv_bridge/build/devel/lib:$LD_LIBRARY_PATH

cd Examples/ROS/ORB_SLAM3
rm -rf build  ## 清除缓存
mkdir build
cd build
# cmake .. -DROS_BUILD_TYPE=Release
#   -DCMAKE_PREFIX_PATH=/home/jiongkesi2/work_space2/catkin_ws/devel;/opt/ros/noetic \
#   -Dcv_bridge_DIR=/home/jiongkesi2/work_space2/catkin_ws/devel/share/cv_bridge/cmake \


cmake \
  -Dcv_bridge_DIR=/home/jiongkesi2/work_space2/catkin_ws/src/vision_opencv/cv_bridge/build/devel/share/cv_bridge/cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0 \
  -DBoost_PYTHON_VERSION=3.8 \
  -DROS_BUILD_TYPE=Release \
  ..

make -j
