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
#include "ORBrefiner.h"
#include "ORBmatcher.h"
#include "Converter.h"
#include <list>

namespace ORB_SLAM2_MapReuse
{

    class Locator
    {

    public:

        Locator(ORBVocabulary *pVoc, KeyFrameDatabase *pKFDB, const string &strSettingsFile, 
                bool bAblation, bool bMatchingUsingBooster, bool bRetrievalUsingBooster, ORBVocabulary *pVocAB);

        cv::Mat visualLocalization(const cv::Mat &im, const double &timestamp);

        void computeForCurrentFrame();

        float CalculateRecall();

        ORBrefiner mORBrefiner;

        list<cv::Mat> mlFramePoses;
        list<double> mlFrameTimes;
        list<bool> mlbFrameLocated;
        list<std::vector<double> > mlCandidateKFTimes;
        list<double> mlRelocKFTimes;

    private:

        bool VisualLocalization();

        ORBVocabulary *mpVocabulary;
        ORBVocabulary *mpVocabularyForBoostingAblation;
        KeyFrameDatabase *mpKeyFrameDatabase;
        ORBextractor *mpORBextractor;
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;
        bool mbRGB;
        bool mbRefineORB;

        int current_image_height;
        int current_image_width;

        Frame mCurrentFrame;

        // Ablation
        bool mbAblation, mbMatchingUsingBooster, mbRetrievalUsingBooster;
    };

}

#endif