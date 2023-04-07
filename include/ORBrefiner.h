//
// Created by antusheng on 3/22/22.
//

#ifndef ORBREFINER_H
#define ORBREFINER_H

#include <vector>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

namespace F = torch::nn::functional;

enum descriptorType {Binary=0, Float=1};

// class ORBrefiner : public torch::nn::Module
// {
// public:
//     ORBrefiner(int in_features, int out_features, descriptorType out_type);

//     torch::Tensor forward(torch::Tensor x);

//     void refine(cv::Mat &descriptors);

// private:
//     int mid_features[2] = {512,512};

//     torch::nn::Sequential network;

//     descriptorType type;
// };

class KeypointEncoderImpl : public torch::nn::Module
{
public:
    KeypointEncoderImpl();

    torch::Tensor forward(torch::Tensor kpts);

private:
    torch::nn::Sequential encoder;
};
TORCH_MODULE(KeypointEncoder);

class DescriptorEncoderImpl : public torch::nn::Module
{
public:
    DescriptorEncoderImpl();

    torch::Tensor forward(torch::Tensor descs);

private:
    torch::nn::Sequential encoder;
};
TORCH_MODULE(DescriptorEncoder);

class HydraAttentionImpl : public torch::nn::Module
{
public:
    HydraAttentionImpl(int d_model);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear query{nullptr};
    torch::nn::Linear key{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
};
TORCH_MODULE(HydraAttention);

class PositionwiseFeedForwardImpl : public torch::nn::Module
{
public:
    PositionwiseFeedForwardImpl(int feature_dim);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential mlp;
    torch::nn::LayerNorm layer_norm{nullptr};
};
TORCH_MODULE(PositionwiseFeedForward);

class AttentionalLayerImpl : public torch::nn::Module
{
public:
    AttentionalLayerImpl(int feature_dim);

    torch::Tensor forward(torch::Tensor x);

private:
    HydraAttention attn{nullptr};
    PositionwiseFeedForward ffn{nullptr};
};
TORCH_MODULE(AttentionalLayer);

class AttentionalNNImpl : public torch::nn::Module
{
public:
    AttentionalNNImpl(int feature_dim, int layer_num);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList layers;
};
TORCH_MODULE(AttentionalNN);

class ORBrefinerImpl : public torch::nn::Module
{
public:
    ORBrefinerImpl();

    torch::Tensor forward(torch::Tensor desc, torch::Tensor kpts);

    void refine(int height, int width, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    KeypointEncoder kenc{nullptr};
    DescriptorEncoder denc{nullptr};
    AttentionalNN attn_proj{nullptr};
    torch::nn::Linear final_proj{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
};
TORCH_MODULE(ORBrefiner);

#endif //ORBREFINER_H
