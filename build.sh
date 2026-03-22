echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW3
rm -rf build 
mkdir build
cd build
cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_PYTHON_VERSION=3.8 \
  -DCMAKE_BUILD_TYPE=Release\
  ..
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."
rm -rf build 
mkdir build
cd build
cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_PYTHON_VERSION=3.8 \
  -DCMAKE_BUILD_TYPE=Release\
  ..
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."
rm -rf build 
mkdir build
cd build
cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_PYTHON_VERSION=3.8 \
  -DCMAKE_BUILD_TYPE=Release\
  ..
make -j

cd ../../../

echo "Uncompress vocabulary ..."



echo "Configuring and building ORB_SLAM3 ..."
rm -rf build 
mkdir build
cd build
cmake \
  -DBOOST_ROOT=/usr \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -DBoost_PYTHON_VERSION=3.8 \
  -DCMAKE_BUILD_TYPE=Release\
  ..
make -j12




