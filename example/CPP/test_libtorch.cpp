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
    cv::Mat descriptors(2000, 32, CV_8UC1);
    auto model = std::make_shared<ORBrefiner>(256, 256, Binary);
    model->eval();
    torch::load(model, "/home/antusheng/Desktop/orb_refine_binary.pt");
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    model->refine(descriptors);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    std::cout << descriptors << std::endl;

    std::cout << "Refine time: " << time << std::endl;

    auto dictRefiner = model->named_parameters();
    for (auto n = dictRefiner.begin(); n != dictRefiner.end(); n++)
    {
        std::cout<<(*n).key()<<std::endl;
    }

    return 0;
}