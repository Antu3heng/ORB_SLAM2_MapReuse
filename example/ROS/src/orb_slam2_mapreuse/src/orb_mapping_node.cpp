/**
 * @file orb_mapping_node.cpp
 * @author Xinjiang Wang (wangxj83@sjtu.edu.cn)
 * @brief ros mapping example 
 * @version 0.1
 * @date 2021-09-09
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <string>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include "System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2_MapReuse::System *pMapper) : mpMapper(pMapper) {}

    void GrabMono(const sensor_msgs::ImageConstPtr &msg);

    void GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight);

    void GrabRGBD(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD);

    ORB_SLAM2_MapReuse::System *mpMapper;
    bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orb_mapping_node");
    ros::NodeHandle nh("~");

    if (argc != 2)
    {
        cerr << endl
             << "Usage: rosrun orb_slam2_mapreuse orb_mapping_node path_to_config" << endl;
        return 1;
    }

    cv::FileStorage fsSettings(argv[1], cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    string image0Topic, image1Topic, vocPath, mapSavePath;
    fsSettings["Camera.image0Topic"] >> image0Topic;
    fsSettings["Vocabulary.Path"] >> vocPath;
    fsSettings["Map.Path"] >> mapSavePath;

    ORB_SLAM2_MapReuse::System::eSensor cameraType;
    int temp = fsSettings["Camera.type"];
    if (temp == 0)
        cameraType = ORB_SLAM2_MapReuse::System::MONOCULAR;
    else if (temp == 1)
    {
        cameraType = ORB_SLAM2_MapReuse::System::STEREO;
        fsSettings["Camera.image1Topic"] >> image1Topic;
    } 
    else if (temp == 2)
    {
        cameraType = ORB_SLAM2_MapReuse::System::RGBD;
        fsSettings["Camera.image1Topic"] >> image1Topic;
    }
    else
    {
        cerr << "Wrong camera type!" << endl;
        return -1;
    }     

    // Create mapping system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2_MapReuse::System Mapper(argv[1], cameraType, ORB_SLAM2_MapReuse::System::SLAM, false);

    ImageGrabber igb(&Mapper);

    ros::Subscriber img_sub;
    message_filters::Subscriber<sensor_msgs::Image> img0_sub(nh, image0Topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> img1_sub(nh, image1Topic, 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), img0_sub, img1_sub);

    switch (cameraType)
    {
        case ORB_SLAM2_MapReuse::System::MONOCULAR:
        {
            ros::Subscriber img_sub = nh.subscribe(image0Topic, 1, &ImageGrabber::GrabMono, &igb);
            break;
        }
        case ORB_SLAM2_MapReuse::System::STEREO:
        {
            fsSettings["Rectification"] >> igb.do_rectify;
            if (igb.do_rectify)
            {
                // Load settings related to stereo calibration
                cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
                fsSettings["LEFT.K"] >> K_l;
                fsSettings["RIGHT.K"] >> K_r;

                fsSettings["LEFT.P"] >> P_l;
                fsSettings["RIGHT.P"] >> P_r;

                fsSettings["LEFT.R"] >> R_l;
                fsSettings["RIGHT.R"] >> R_r;

                fsSettings["LEFT.D"] >> D_l;
                fsSettings["RIGHT.D"] >> D_r;

                int rows_l = fsSettings["LEFT.height"];
                int cols_l = fsSettings["LEFT.width"];
                int rows_r = fsSettings["RIGHT.height"];
                int cols_r = fsSettings["RIGHT.width"];

                if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                    rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
                {
                    cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
                    return -1;
                }

                cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, igb.M1l, igb.M2l);
                cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, igb.M1r, igb.M2r);
            }
            sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));
            break;
        }
        case ORB_SLAM2_MapReuse::System::RGBD:
        {
            sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2));
            break;
        }
        default:
            break;
    }

    ros::spin();

    // Stop all threads
    Mapper.Shutdown();

    // Save the map
    // Mapper.SaveMap(mapSavePath);
    Mapper.SaveMapUsingBoost(mapSavePath); // using boost save the map

    return 0;
}

void ImageGrabber::GrabMono(const sensor_msgs::ImageConstPtr &msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpMapper->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (do_rectify)
    {
        cv::Mat imLeft, imRight;
        cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
        mpMapper->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
    }
    else
    {
        mpMapper->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image, cv_ptrLeft->header.stamp.toSec());
    }
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpMapper->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, cv_ptrRGB->header.stamp.toSec());
}