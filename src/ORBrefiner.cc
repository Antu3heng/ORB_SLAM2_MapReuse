//
// Created by antusheng on 3/22/22.
//

#include "ORBrefiner.h"
#include <chrono>

// ORBrefiner::ORBrefiner(int in_features, int out_features, descriptorType out_type)
// {
//     network->push_back(torch::nn::Linear(torch::nn::LinearOptions(in_features, mid_features[0])));
//     network->push_back(torch::nn::BatchNorm1d(mid_features[0]));
//     network->push_back(torch::nn::ELU());
//     network->push_back(torch::nn::Linear(torch::nn::LinearOptions(mid_features[0], mid_features[1])));
//     network->push_back(torch::nn::BatchNorm1d(mid_features[1]));
//     network->push_back(torch::nn::ELU());
//     network->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(0.3)));
//     network->push_back(torch::nn::Linear(torch::nn::LinearOptions(mid_features[1], out_features)));

//     network = register_module("network", network);

//     type = out_type;
// }

// torch::Tensor ORBrefiner::forward(torch::Tensor x)
// {
//     x = network->forward(x);
//     if (type == Binary)
//     {
//         x = torch::tanh(x);
//         x = (x >= 0).to(torch::kByte);
//     }
//     if (type == Float)
//         x = F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1));

//     return x;
// }

// void ORBrefiner::refine(cv::Mat &descriptors)
// {
//     torch::TensorOptions grad_false(torch::requires_grad(false));
//     // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//     cv::Mat tmps(descriptors.rows, descriptors.cols * 8, CV_8UC1);
//     for (int i = 0; i < descriptors.rows; i++)
//     {
//         uchar* descriptor = descriptors.ptr(i);
//         uchar* tmp = tmps.ptr(i);
//         for (int j = 0; j < descriptors.cols; j++)
//         {
//             uchar val = descriptor[j];
//             for (int k = 0; k < 8; k++)
//                 tmp[8 * j + k] = ((val >> k) & 1);
//         }
//     }
//     auto in = torch::from_blob(tmps.data, {tmps.rows, tmps.cols}, torch::kByte);
//     in = in.to(torch::kFloat32);
//     // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//     // double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//     // std::cout << "convert mat to tensor: " << time << std::endl;
//     auto out = this->forward(in);
//     // t1 = std::chrono::steady_clock::now();
//     tmps = cv::Mat(descriptors.rows, descriptors.cols * 8, CV_8UC1, out.data_ptr<uchar>());
//     for (int i = 0; i < descriptors.rows; i++)
//     {
//         uchar* descriptor = descriptors.ptr(i);
//         uchar* tmp = tmps.ptr(i);
//         for (int j = 0; j < descriptors.cols; j++)
//         {
//             int val = 0;
//             for (int k = 0; k < 8; k++)
//                 val |= tmp[8 * j + k] << k;
//             descriptor[j] = (uchar)val;
//         }
//     }
//     // t2 = std::chrono::steady_clock::now();
//     // time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//     // std::cout << "convert tensor to mat: " << time << std::endl;
// }

KeypointEncoderImpl::KeypointEncoderImpl()
{
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(4, 32)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(32, 64)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(64, 128)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 256)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 256)));
    encoder = register_module("encoder", encoder);
}

torch::Tensor KeypointEncoderImpl::forward(torch::Tensor kpts)
{
    return encoder->forward(kpts);
}

DescriptorEncoderImpl::DescriptorEncoderImpl()
{
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 512)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 256)));
    encoder->push_back(torch::nn::ReLU());
    encoder->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, 256)));
    encoder = register_module("encoder", encoder);
}

torch::Tensor DescriptorEncoderImpl::forward(torch::Tensor descs)
{
    auto residual = descs;
    return residual + encoder->forward(descs);
}

HydraAttentionImpl::HydraAttentionImpl(int d_model)
{
    query = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
    key = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
    value = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
    proj = register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
    layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(1e-6)));
}

torch::Tensor HydraAttentionImpl::forward(torch::Tensor x)
{
    torch::Tensor residual = x;
    torch::Tensor q = query(x);
    torch::Tensor k = key(x);
    torch::Tensor v = value(x);
    q = torch::sigmoid(q);
    k = torch::softmax(k.t(), 1).t();
    torch::Tensor kv = (k * v).sum(-2, true);
    x = q * kv;
    x = proj(x);
    x += residual;
    x = layer_norm(x);
    return x;
}

PositionwiseFeedForwardImpl::PositionwiseFeedForwardImpl(int feature_dim)
{
    mlp->push_back(torch::nn::Linear(torch::nn::LinearOptions(feature_dim, feature_dim * 2)));
    mlp->push_back(torch::nn::ReLU());
    mlp->push_back(torch::nn::Linear(torch::nn::LinearOptions(feature_dim * 2, feature_dim)));
    mlp = register_module("mlp", mlp);
    layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({feature_dim}).eps(1e-6)));
}

torch::Tensor PositionwiseFeedForwardImpl::forward(torch::Tensor x)
{
    torch::Tensor residual = x;
    x = mlp->forward(x);
    x += residual;
    x = layer_norm(x);
    return x;
}

AttentionalLayerImpl::AttentionalLayerImpl(int feature_dim)
{
    attn = HydraAttention(feature_dim);
    ffn = PositionwiseFeedForward(feature_dim);
    attn = register_module("attn", attn);
    ffn = register_module("ffn", ffn);
}

torch::Tensor AttentionalLayerImpl::forward(torch::Tensor x)
{
    x = attn->forward(x);
    x = ffn->forward(x);
    return x;
}

AttentionalNNImpl::AttentionalNNImpl(int feature_dim, int layer_num)
{
    for (int i = 0; i < layer_num; i++)
        layers->push_back(AttentionalLayer(feature_dim));
    layers = register_module("layers", layers);
}

torch::Tensor AttentionalNNImpl::forward(torch::Tensor x)
{
    for (const auto &layer : *layers)
    {
        x = layer->as<AttentionalLayer>()->forward(x);
    }
    return x;
}

ORBrefinerImpl::ORBrefinerImpl()
{
    kenc = KeypointEncoder();
    denc = DescriptorEncoder();
    attn_proj = AttentionalNN(256, 4);
    final_proj = torch::nn::Linear(torch::nn::LinearOptions(256, 256));
    layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({256}).eps(1e-6));

    kenc = register_module("kenc", kenc);
    denc = register_module("denc", denc);
    attn_proj = register_module("attn_proj", attn_proj);
    final_proj = register_module("final_proj", final_proj);
    layer_norm = register_module("layer_norm", layer_norm);
}

torch::Tensor ORBrefinerImpl::forward(torch::Tensor desc, torch::Tensor kpts)
{
    torch::Tensor x = denc->forward(desc) + kenc->forward(kpts);
    x = layer_norm(x);
    x = attn_proj(x);
    x = final_proj(x);
    x = (x >= 0).to(torch::kByte);
    return x;
}

void ORBrefinerImpl::refine(int height, int width, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    torch::TensorOptions grad_false(torch::requires_grad(false));
    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // prepare for the keypoint data
    float x0 = (float)width / 2.0;
    float y0 = (float)height / 2.0;
    float scale = (float)std::max(width, height) * 0.7;
    std::vector<float> tmp_kpts;
    const float factorPI = (float)(CV_PI/180.f);
    for (auto &keypoint : keypoints)
    {
        tmp_kpts.emplace_back((keypoint.pt.x - x0) / scale);
        tmp_kpts.emplace_back((keypoint.pt.y - y0) / scale);
        tmp_kpts.emplace_back(keypoint.size / 31.0);
        tmp_kpts.emplace_back(keypoint.angle * factorPI);
    }
    auto kpts = torch::from_blob(tmp_kpts.data(), {(int)keypoints.size(), 4}, torch::kFloat32).clone();
    if (torch::cuda::is_available())
        kpts = kpts.to(torch::kCUDA);
    // prepare for the descriptor data
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
        descs = descs.to(torch::kCUDA);
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // std::cout << "convert mat to tensor: " << time << std::endl;
    auto out = this->forward(descs, kpts).to(torch::kCPU);
    // t1 = std::chrono::steady_clock::now();
    auto tmps = cv::Mat(descriptors.rows, descriptors.cols * 8, CV_8UC1, out.data_ptr<uchar>());
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