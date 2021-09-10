/**
 * @file Locator.h
 * @author Xinjiang Wang (wangxj83@sjtu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-05-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef LOCATOR_H
#define LOCATOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Converter.h"
#include <list>

namespace ORB_SLAM2_MapReuse
{

    class Locator
    {

    public:

        Locator(ORBVocabulary *pVoc, KeyFrameDatabase *pKFDB, const string &strSettingsFile);

        cv::Mat visualLocalization(const cv::Mat &im, const double &timestamp);

        float CalculateRecall();

        list<cv::Mat> mlFramePoses;
        list<double> mlFrameTimes;
        list<bool> mlbFrameLocated;

    private:

        bool VisualLocalization();

        ORBVocabulary *mpVocabulary;
        KeyFrameDatabase *mpKeyFrameDatabase;
        ORBextractor *mpORBextractor;
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;
        bool mbRGB;

        Frame mCurrentFrame;
    };

}

#endif