//
// Created by antusheng on 3/22/22.
//

#include <iostream>
#include <chrono>
#include <memory>
#include <torch/torch.h>
#include "ORBrefiner.h"

int main()
{   
    cv::RNG rnger(cv::getTickCount());
    // keypoints
    cv::Mat cv_keypoints;
    cv_keypoints.create(2000, 4, CV_32FC1);
    rnger.fill(cv_keypoints, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(720.));
    std::vector<cv::KeyPoint> keypoints;
    keypoints.resize(2000);
    for (int i = 0; i < 2000; i++)
    {
        keypoints[i].pt.x = cv_keypoints.at<float>(i, 0);
        keypoints[i].pt.y = cv_keypoints.at<float>(i, 1);
        keypoints[i].size = cv_keypoints.at<float>(i, 2);
        keypoints[i].angle = cv_keypoints.at<float>(i, 3);
    }
    // descriptors
    cv::Mat descriptors;
    descriptors.create(2000, 32, CV_8UC1);
    rnger.fill(descriptors, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
    auto model = ORBrefiner();
    torch::load(model, "/home/drone/wxj/myCode/ORB_SLAM2_MapReuse/checkpoints/orbslam2-binary.pt");
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available!" << std::endl;
        model->to(torch::kCUDA);
    }
    else
        std::cout << "CUDA is not available!" << std::endl;
    model->eval();
    double total_time = 0;
    for (int i = 0; i < 1000; i++)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        model->refine(720, 720, keypoints, descriptors);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    }

    std::cout << descriptors << std::endl;

    std::cout << "Refine time: " << total_time / 1000. << std::endl;

    auto dictRefiner = model->named_parameters();
    for (auto n = dictRefiner.begin(); n != dictRefiner.end(); n++)
    {
        std::cout<<(*n).key()<<std::endl;
    }

    return 0;
}