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


```
# 🚀 2. Running Examples (Native C++)
```
Monocular-Inertial (EuRoC Dataset)
./Examples/Monocular-Inertial/mono_inertial_euroc \
    Vocabulary/voc_binary_tartan_8u_6.bin \
    Examples/Monocular-Inertial/EuRoC.yaml \
    <PATH_TO_DATASET>/EuRoC/V2_03 \
    <YOUR_PATH>/Examples/Monocular-Inertial/EuRoC_TimeStamps/V2_03.txt


Stereo-Inertial (EuRoC Dataset)
./Examples/Stereo-Inertial/stereo_inertial_euroc \
    Vocabulary/voc_binary_tartan_8u_6.bin \
    Examples/Stereo-Inertial/EuRoC.yaml \
    <PATH_TO_DATASET>/EuRoC/V2_03 \
    <YOUR_PATH>/Examples/Stereo-Inertial/EuRoC_TimeStamps/V203.txt


```
# 🤖 3. Running with ROS
```
Step 1: Start ROS Master
roscore

Step 2: Run SLAM Node
export ROS_PACKAGE_PATH=<YOUR_PROJECT_PATH>/Examples/ROS:$ROS_PACKAGE_PATH

# For UUV Simulation Dataset (Using SuperPoint-based Vocabulary)
rosrun ORB_SLAM3 Mono_Inertial ./Vocabulary/6_superpoint_uuv_voc_6layer_K10_1_CV4.bin Examples/Monocular-Inertial/UUV_slam_new.yaml

Step 3: Play Bag File

# Example: USS-2 Underwater Sequence
rosbag play <PATH_TO_BAG>/uss-2.bag \
    /D435i_camera/color/image_raw:=/camera/color/image_raw \
    /example_auv/imu:=/camera/imu

```
# 📊 4. Evaluation
```

We use evo for trajectory accuracy assessment.
# UUV Dataset Evaluation (TUM Format)
evo_ape tum uss-2.tum CameraTrajectory.txt -s --align --correct_scale --plot --plot_mode xyz -v --t_offset -0.02


```
# 🤝 Acknowledgments
```
Our work is built upon and inspired by the following excellent open-source projects:

ORB_SLAM3

Rover-SLAM

SP-Loop



```
# ✍️ Citation
```
@ARTICLE{11408052,
  author={Wang, Jiongming and Wang, Zixuan and Shi, Yefan and Zhang, Wei},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Underwater-Robust SLAM and a Simulation Dataset for UUV SLAM}, 
  year={2026},
  volume={75},
  number={},
  pages={1-14},
  keywords={Simultaneous localization and mapping;Feature extraction;Location awareness;Visualization;Optimization;Tracking loops;Trajectory;Computer architecture;Computational modeling;Accuracy;Simultaneous localization and mapping (SLAM);underwater environment;unmanned underwater vehicle (UUV);visual–inertial odometry},
  doi={10.1109/TIM.2026.3666037}}









