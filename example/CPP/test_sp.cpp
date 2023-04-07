//
// Created by antusheng on 3/22/22.
//

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

int main()
{   
    cv::RNG rnger(cv::getTickCount());

    // change this
    std::string checkpoint_path = "/home/drone/wxj/myCode/ORB_SLAM2_MapReuse/checkpoints/";
    std::vector<std::string> names{"superpoint-float-aft.pt", "superpoint-float-trans.pt"};
    for (auto name : names)
    {
        auto model = torch::jit::load(checkpoint_path + name);
        if (torch::cuda::is_available())
        {
            std::cout << "CUDA is available!" << std::endl;
            model.to(torch::kCUDA);
        }
        else
            std::cout << "CUDA is not available!" << std::endl;
        model.eval();

        std::vector<int> nums{500, 1000, 2000, 4000, 8000};
        for (auto num : nums)
        {
            // keypoints
            cv::Mat cv_keypoints;
            cv_keypoints.create(num, 3, CV_32FC1);
            rnger.fill(cv_keypoints, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(1.));
            auto kpts = torch::from_blob(cv_keypoints.data, {cv_keypoints.rows, cv_keypoints.cols}, torch::kFloat32).clone();
            // descriptors
            cv::Mat descriptors;
            descriptors.create(num, 32, CV_8UC1);
            rnger.fill(descriptors, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
            cv::Mat tmp_descs(descriptors.rows, descriptors.cols * 8, CV_8UC1);
            for (int i = 0; i < descriptors.rows; i++)
            {
                uchar* descriptor = descriptors.ptr(i);
                uchar* tmp = tmp_descs.ptr(i);
                for (int j = 0; j < descriptors.cols; j++)
                {
                    uchar val = descriptor[j];
                    for (int k = 0; k < 8; k++)
                        tmp[8 * j + k] = ((val >> k) & 1);
                }
            }
            auto descs = torch::from_blob(tmp_descs.data, {tmp_descs.rows, tmp_descs.cols}, torch::kByte).clone();
            descs = descs.to(torch::kFloat32);
            
            if (torch::cuda::is_available())
            {
                kpts = kpts.to(torch::kCUDA);
                descs = descs.to(torch::kCUDA);
            }  
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(descs);
            inputs.emplace_back(kpts);

            // for (int i = 0; i < 50; i++)
            // {
            //     model.forward(inputs).toTensor().to(torch::kCPU);
            // }

            double total_time = 0;
            for (int i = 0; i < 1000; i++)
            {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                model.forward(inputs).toTensor().to(torch::kCPU);
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            }

            std::cout << std::endl << name << std::endl << num << " feature boost time: " << total_time / 1000. << std::endl;
        }
    }

    return 0;
}