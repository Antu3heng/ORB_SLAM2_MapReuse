//
// Created by antusheng on 3/22/22.
//

#include "ORBrefiner.h"
#include <chrono>

ORBrefiner::ORBrefiner(int in_features, int out_features, descriptorType out_type)
{
    network->push_back(torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_features[0])));
    network->push_back(torch::nn::BatchNorm1d(mid_features[0]));
    network->push_back(torch::nn::ELU());
    network->push_back(torch::nn::Linear(torch::nn::LinearOptions(mid_features[0], mid_features[1])));
    network->push_back(torch::nn::BatchNorm1d(mid_features[1]));
    network->push_back(torch::nn::ELU());
    network->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.3)));
    network->push_back(torch::nn::Linear(torch::nn::LinearOptions(mid_features[1], out_features)));

    network = register_module("network", network);

    type = out_type;
}

torch::Tensor ORBrefiner::forward(torch::Tensor x)
{
    x = network->forward(x);
    if (type == Binary)
    {
        x = torch::tanh(x);
        x = (x >= 0).to(torch::kByte);
    }
    if (type == Float)
        x = F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1));

    return x;
}

void ORBrefiner::refine(cv::Mat &descriptors)
{
    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::Mat tmps(descriptors.rows, descriptors.cols * 8, CV_8UC1);
    for (int i = 0; i < descriptors.rows; i++)
    {
        uchar* descriptor = descriptors.ptr(i);
        uchar* tmp = tmps.ptr(i);
        for (int j = 0; j < descriptors.cols; j++)
        {
            uchar val = descriptor[j];
            for (int k = 0; k < 8; k++)
                tmp[8 * j + k] = ((val >> k) & 1);
        }
    }
    auto in = torch::from_blob(tmps.data, {tmps.rows, tmps.cols}, torch::kByte);
    in = in.to(torch::kFloat32);
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // std::cout << "convert mat to tensor: " << time << std::endl;
    auto out = this->forward(in);
    // t1 = std::chrono::steady_clock::now();
    tmps = cv::Mat(descriptors.rows, descriptors.cols * 8, CV_8UC1, out.data_ptr<uchar>());
    for (int i = 0; i < descriptors.rows; i++)
    {
        uchar* descriptor = descriptors.ptr(i);
        uchar* tmp = tmps.ptr(i);
        for (int j = 0; j < descriptors.cols; j++)
        {
            int val = 0;
            for (int k = 0; k < 8; k++)
                val |= tmp[8 * j + k] << k;
            descriptor[j] = (uchar)val;
        }
    }
    // t2 = std::chrono::steady_clock::now();
    // time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // std::cout << "convert tensor to mat: " << time << std::endl;
}
