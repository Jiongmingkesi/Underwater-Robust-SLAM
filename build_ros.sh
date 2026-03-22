echo "Building ROS nodes"


cd Examples/ROS/ORB_SLAM3
rm -rf build  ## 清除缓存
mkdir build
cd build


cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_PYTHON_VERSION=3.8 \
  -DROS_BUILD_TYPE=Release \
  ..

make -j
