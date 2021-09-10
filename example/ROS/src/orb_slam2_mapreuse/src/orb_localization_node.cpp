/**
 * @file orb_localization_node.cpp
 * @author Xinjiang Wang (wangxj83@sjtu.edu.cn)
 * @brief ros localization example 
 * @version 0.1
 * @date 2021-09-09
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <fstream>
#include <string>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include "System.h"
#include"Converter.h"

using namespace std;

ros::Publisher pose_pub;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2_MapReuse::System *pLocalization) : mpLocalization(pLocalization) {}

    void GrabImage(const sensor_msgs::ImageConstPtr &msg);

    ORB_SLAM2_MapReuse::System *mpLocalization;
    bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orb_localization_node");
    ros::NodeHandle nh("~");

    if(argc != 2)
    {
        cerr << endl << "Usage: rosrun orb_slam2_mapreuse orb_localization_node path_to_config" << endl;
        return 1;
    }

    cv::FileStorage fsSettings(argv[1], cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    string image0Topic, trajSavePath, mapSavePath;
    fsSettings["Camera.image0Topic"] >> image0Topic;
    fsSettings["Map.Path"] >> mapSavePath;
    fsSettings["Trajectory.SavePath"] >> trajSavePath;

    ORB_SLAM2_MapReuse::System::eSensor cameraType;
    int temp = fsSettings["Camera.type"];
    if (temp == 0)
        cameraType = ORB_SLAM2_MapReuse::System::MONOCULAR;
    else if (temp == 1)
    {
        cameraType = ORB_SLAM2_MapReuse::System::STEREO;
    } 
    else if (temp == 2)
    {
        cameraType = ORB_SLAM2_MapReuse::System::RGBD;
    }
    else
    {
        cerr << "Wrong camera type!" << endl;
        return -1;
    }

    // Create Localization system. 
    ORB_SLAM2_MapReuse::System Localization(argv[1], cameraType, ORB_SLAM2_MapReuse::System::VisualLocalization);
    Localization.LoadMap(mapSavePath);
    // Localization.LoadMapUsingBoost(mapSavePath); // using boost load the map

    ImageGrabber igb(&Localization);

    if (cameraType == ORB_SLAM2_MapReuse::System::STEREO)
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
    }

    ros::Subscriber img_sub = nh.subscribe(image0Topic, 10, &ImageGrabber::GrabImage, &igb);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mapLoc/pose", 10);

    ros::spin();

    // Stop all threads
    Localization.Shutdown();

    // Save the trajectory
    Localization.SaveTrajectoryTUM(trajSavePath + "FrameTrajectory_Localization.txt");

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
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

    cv::Mat Tcw;

    if (do_rectify)
    {
        cv::Mat imLeft;
        cv::remap(cv_ptr->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
        Tcw = mpLocalization->ORBLocalization(imLeft, cv_ptr->header.stamp.toSec());
    }
    else
        Tcw = mpLocalization->ORBLocalization(cv_ptr->image, cv_ptr->header.stamp.toSec());

    if(!cv::countNonZero(Tcw) < 1)
    {
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        vector<float> q = ORB_SLAM2_MapReuse::Converter::toQuaternion(Rwc);

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time(msg->header.stamp.toSec());
        pose.header.frame_id = "world";
        pose.pose.orientation.x = q[0];
        pose.pose.orientation.y = q[1];
        pose.pose.orientation.z = q[2];
        pose.pose.orientation.w = q[3];
        pose.pose.position.x = twc.at<float>(0);
        pose.pose.position.y = twc.at<float>(1);
        pose.pose.position.z = twc.at<float>(2);
        pose_pub.publish(pose);

        // cout << fixed;
        // cout << setprecision(6) << cv_ptr->header.stamp << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }   
}