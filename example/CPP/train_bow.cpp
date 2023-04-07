#include <iostream>
#include <chrono>
#include <memory>
#include <torch/torch.h>
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "ORBrefiner.h"

using namespace std;
using namespace ORB_SLAM2_MapReuse;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: trainBow path_to_image_folder k l" << endl;
        return 1;
    }

    // find the images
    cout << endl << "Finding images ... " << endl;
    string image_path(argv[1]);
    cv::String pattern_png = image_path + "*.png";
    vector<cv::String> png_image_names;
    cv::glob(pattern_png, png_image_names, true);
    cv::String pattern_jpg = image_path + "*.jpg";
    vector<cv::String> jpg_image_names;
    cv::glob(pattern_jpg, jpg_image_names, true);
    cout << endl << "Found " << png_image_names.size() + jpg_image_names.size() << " images!" << endl;

    // read the image and detect the ORB features
    cout << endl << "Reading images and detecting ORB features ... " << endl;
    // ORB Extractor
    ORBextractor *pORBextractor = new ORBextractor(2000, 1.2, 8, 20, 7);
    // ORB Refiner
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
    vector<vector<cv::Mat> > features;
    for (int i = 0; i < png_image_names.size(); i++)
    {
        // if (i >= 50000)
        //     break;
        cout << png_image_names[i] << endl;
        cv::Mat image = cv::imread(png_image_names[i], 0);
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        (*pORBextractor)(image, cv::Mat(), keypoints, descriptors);
        if (keypoints.size() <= 50)
            continue;
        model->refine(image.rows, image.cols, keypoints, descriptors);
        features.push_back(vector<cv::Mat>());
        changeStructure(descriptors, features.back());
    }
    for (int i = 0; i < jpg_image_names.size(); i++)
    {
        // if (i >= 50000)
        //     break;
        cout << jpg_image_names[i] << endl;
        cv::Mat image = cv::imread(jpg_image_names[i], 0);
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        (*pORBextractor)(image, cv::Mat(), keypoints, descriptors);
        if (keypoints.size() <= 50)
            continue;
        model->refine(image.rows, image.cols, keypoints, descriptors);
        features.push_back(vector<cv::Mat>());
        changeStructure(descriptors, features.back());
    }
    // for (const auto &image_name : png_image_names)
    // {
    //     cv::Mat image = cv::imread(image_name, 0);
    //     vector<cv::KeyPoint> keypoints;
    //     cv::Mat descriptors;
    //     (*pORBextractor)(image, cv::Mat(), keypoints, descriptors);
    //     if (keypoints.size() <= 50)
    //         continue;
    //     model->refine(image.rows, image.cols, keypoints, descriptors);
    //     features.push_back(vector<cv::Mat>());
    //     changeStructure(descriptors, features.back());
    // }

    // create the vocabulary
    int k = atoi(argv[2]);
    int l = atoi(argv[3]);
    DBoW2::WeightingType weight = DBoW2::TF_IDF;
    DBoW2::ScoringType scoring = DBoW2::L1_NORM;
    ORBVocabulary orbVoc(k, l, weight, scoring);
    cout << endl << "Creating a " << k << "^" << l << " ORB vocabulary..." << endl;
    orbVoc.create(features);
    cout << "Done!" << endl;
    cout << "Vocabulary information: " << endl << orbVoc << endl;

    // save the vocabulary
    cout << endl << "Saving vocabulary... " << endl;
    string vocDir("voc.txt");
    orbVoc.saveToTextFile(vocDir);
    cout << "Done!" << endl;

    return 0;
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}