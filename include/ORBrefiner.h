//
// Created by antusheng on 3/22/22.
//

#ifndef ORBREFINER_H
#define ORBREFINER_H

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

namespace F = torch::nn::functional;

enum descriptorType {Binary=0, Float=1};

class ORBrefiner : public torch::nn::Module
{
public:
    ORBrefiner(int in_features, int out_features, descriptorType out_type);

    torch::Tensor forward(torch::Tensor x);

    void refine(cv::Mat &descriptors);

private:
    int mid_features[2] = {512,512};

    torch::nn::Sequential network;

    descriptorType type;
};

#endif //ORBREFINER_H
