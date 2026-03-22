
#include "System.h" 


void RunLSTMCorrection(ORB_SLAM3::System *pSLAM)
{
    
    int N_STEP = 200;
    std::vector<std::pair<double, Sophus::SE3f>> trajectory_data;

    trajectory_data = pSLAM->GetRecentOptimizedTrajectory(N_STEP);

    
    if (trajectory_data.size() < N_STEP)
    {
        
        

        
        
        std::cout << "Data not enough for LSTM yet: " << trajectory_data.size() << std::endl;
        return;
    }

    
    
    std::vector<float> lstm_input;
    for (auto &item : trajectory_data)
    {
        double timestamp = item.first;
        Sophus::SE3f &pose = item.second;

        Eigen::Vector3f t = pose.translation();
        Eigen::Quaternionf q = pose.unit_quaternion();

        lstm_input.push_back(t.x());
        lstm_input.push_back(t.y());
        lstm_input.push_back(t.z());
        
    }

    
}