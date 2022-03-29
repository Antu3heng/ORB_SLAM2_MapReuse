//
// Created by antusheng on 3/22/22.
//

#include "ORBrefiner.h"

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
        x = (x >= 0).toType(torch::kInt8);
    }
    if (type == Float)
        x = F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1));

    return x;
}

void ORBrefiner::refine(cv::Mat &descriptors)
{
    torch::Tensor in = torch::empty({descriptors.rows, descriptors.cols * 8}, torch::kInt8);
    for (int i = 0; i < descriptors.rows; i++)
    {
        uchar* descriptor = descriptors.ptr(i);
        for (int j = 0; j < descriptors.cols; j++)
        {
            uchar val = descriptor[j];
            for (int k = 0; k < 8; k++)
                in[i][8 * j + k] = (uchar)((val >> k) & 1);
        }
    }
    auto res = this->forward(in);
    for (int i = 0; i < descriptors.rows; i++)
    {
        uchar* descriptor = descriptors.ptr(i);
        for (int j = 0; j < descriptors.cols; j++)
        {
            int val;
            for (int k = 0; k < 8; k++)
                val |= res[i][8 * j + k].item<int>() << k;
            descriptor[j] = (uchar)val;
        }
    }
}
