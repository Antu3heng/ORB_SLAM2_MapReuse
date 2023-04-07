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

torch::jit::script::Module model;

void build_model(std::string path)
{
    model = torch::jit::load(path);
    if (torch::cuda::is_available())
        model.to(torch::kCUDA);
    model.eval();
}

void boost(int height, int width, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

int main()
{   
    cv::RNG rnger(cv::getTickCount());

    // change this
    std::string checkpoint_path = "/home/drone/wxj/myCode/ORB_SLAM2_MapReuse/checkpoints/";
    std::vector<std::string> names{"orb-binary-aft.pt", "orb-binary-trans.pt"};
    for (auto name : names)
    {
        std::vector<int> nums{100, 500, 1000};
        for (auto num : nums)
        {
            // model
            build_model(checkpoint_path + name);

            // keypoints
            cv::Mat cv_keypoints;
            cv_keypoints.create(num, 4, CV_32FC1);
            rnger.fill(cv_keypoints, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(720.));
            std::vector<cv::KeyPoint> keypoints;
            keypoints.resize(num);
            for (int i = 0; i < num; i++)
            {
                keypoints[i].pt.x = cv_keypoints.at<float>(i, 0);
                keypoints[i].pt.y = cv_keypoints.at<float>(i, 1);
                keypoints[i].size = cv_keypoints.at<float>(i, 2);
                keypoints[i].angle = cv_keypoints.at<float>(i, 3);
            }

            // descriptors
            cv::Mat descriptors;
            descriptors.create(num, 32, CV_8UC1);
            rnger.fill(descriptors, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));

            float x0 = (float)720 / 2.0;
            float y0 = (float)720 / 2.0;
            float scale = (float)std::max(720, 720) * 0.7;
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
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(descs);
            inputs.emplace_back(kpts);

            // warm-up
            for (int i = 0; i < 50; i++)
            {
                model.forward(inputs).toTensor().to(torch::kCPU);
            }

            double total_time = 0;
            for (int i = 0; i < 1000; i++)
            {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                model.forward(inputs).toTensor().to(torch::kCPU);
                // boost(720, 720, keypoints, descriptors);
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            }

            std::cout << std::endl << name << std::endl << num << " feature boost time: " << total_time / 1000. << std::endl;
        }
    }

    return 0;
}

void boost(int height, int width, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // pre-process
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
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(descs);
    inputs.emplace_back(kpts);
    // process
    auto out = model.forward(inputs).toTensor().to(torch::kByte).to(torch::kCPU);
    // post-process
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
}