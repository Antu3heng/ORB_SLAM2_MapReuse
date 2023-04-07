//
// Created by antusheng on 3/22/22.
//

#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "SuperPoint.h"

int main()
{   
    auto model = std::make_shared<ORB_SLAM2::SuperPoint>();
    torch::load(model, "/home/drone/wxj/myCode/ORB_SLAM2_MapReuse/checkpoints/sp.pt");

    cv::Mat img = cv::imread("/home/drone/wxj/myCode/ORB_SLAM2_MapReuse/example/CPP/img.jpg", 0);

    double total_time = 0;
    for (int i = 0; i < 10; i++)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        descriptors = ORB_SLAM2::SPdetect(model, img, keypoints, 0.015, false, true);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    }

    std::cout << "SP extract time: " << total_time / 10. << std::endl;

    return 0;
}