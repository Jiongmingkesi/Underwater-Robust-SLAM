# Underwater-Robust SLAM and a Simulation Dataset for UUV SLAM

This repository contains the official implementation of **Underwater-Robust SLAM** and a high-fidelity simulation dataset for unmanned underwater vehicle (UUV) visual-inertial SLAM.

- 📄 **Paper**: [Underwater-Robust SLAM and a Simulation Dataset for UUV SLAM](https://ieeexplore.ieee.org/document/11408052/)
- 💻 **Code**: Publicly available (Modified for OpenCV 4 & ROS)
- 🌊 **Dataset**: Specialized underwater simulation data included

---

# 📥 0. Pre-trained Models & Vocabulary
> **Important**: The model and vocabulary files are too large for standard git storage. 

Please download `model_and_other_files.zip` (1.1GB) from our **[Releases Page](https://github.com/Jiongmingkesi/Underwater-Robust-SLAM/releases)**. 
Extract the contents to the project root directory before running the examples.

---

# 🛠 1. Installation & Compilation
> Ensure you have the necessary dependencies installed (OpenCV 4, Eigen 3, Pangolin, etc.).

### Build the Core System
```bash
chmod +x build.sh  
./build.sh


Build ROS Wrapper

# Replace <YOUR_PROJECT_PATH> with your actual directory
export ROS_PACKAGE_PATH=<YOUR_PROJECT_PATH>/Examples/ROS:$ROS_PACKAGE_PATH

# Verify the ROS package
rospack find ORB_SLAM3

./build_ros.sh


'''
#🚀 2. Running Examples (Native C++)
'''
```bash
## 🚀 2. Running Examples (Native C++)
./Examples/Monocular-Inertial/mono_inertial_euroc ...










