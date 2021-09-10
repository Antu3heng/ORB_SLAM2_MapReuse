/**
 * @file Locator.cc
 * @author Xinjiang Wang (wangxj83@sjtu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-05-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "Locator.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include <opencv2/opencv.hpp>

using namespace std;

namespace ORB_SLAM2_MapReuse
{

    Locator::Locator(ORBVocabulary *pVoc, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingsFile)
            : mpVocabulary(pVoc), mpMap(pMap), mpKeyFrameDatabase(pKFDB)
    {
        cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
        float fx = fsSettings["Camera.fx"];
        float fy = fsSettings["Camera.fy"];
        float cx = fsSettings["Camera.cx"];
        float cy = fsSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fsSettings["Camera.k1"];
        DistCoef.at<float>(1) = fsSettings["Camera.k2"];
        DistCoef.at<float>(2) = fsSettings["Camera.p1"];
        DistCoef.at<float>(3) = fsSettings["Camera.p2"];
        const float k3 = fsSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fsSettings["Camera.bf"];

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;

        fsSettings["Camera.RGB"] >> mbRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fsSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fsSettings["ORBextractor.scaleFactor"];
        int nLevels = fsSettings["ORBextractor.nLevels"];
        int fIniThFAST = fsSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fsSettings["ORBextractor.minThFAST"];

        mpORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    }

    cv::Mat Locator::visualLocalization(const cv::Mat &im, const double &timestamp)
    {
        cv::Mat ImGray = im;

        if (ImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(ImGray, ImGray, CV_RGB2GRAY);
            else
                cvtColor(ImGray, ImGray, CV_BGR2GRAY);
        } else if (ImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(ImGray, ImGray, CV_RGBA2GRAY);
            else
                cvtColor(ImGray, ImGray, CV_BGRA2GRAY);
        }
        mCurrentFrame = Frame(ImGray, timestamp, mpORBextractor, mpVocabulary, mK, mDistCoef, mbf, 0.0f);

        if (VisualLocalization())
        {
            mlFramePoses.push_back(mCurrentFrame.mTcw);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbFrameLocated.push_back(true);
        } else
        {
            mCurrentFrame.mTcw = cv::Mat::zeros(4, 4, CV_32F);
            mlFramePoses.push_back(mCurrentFrame.mTcw);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbFrameLocated.push_back(false);
        }

        return mCurrentFrame.mTcw.clone();
    }

    bool Locator::VisualLocalization()
    {
        mCurrentFrame.ComputeBoW();
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++)
        {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                } else
                {
                    auto *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                    continue;

                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                if (!Tcw.empty())
                {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(nullptr);

                    if (nGood < 50)
                    {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50)
                                {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = nullptr;
                                }
                            }
                        }
                    }

                    if (nGood >= 50)
                    {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
            return false;
        else
            return true;
    }

    void Locator::SetMap(Map *pMap)
    {
        mpMap = pMap;
    }

    float Locator::CalculateRecall()
    {
        int count = 0;

        for (bool &it: mlbFrameLocated)
            if (it)
                count++;

        std::cout << endl << "Total " << mlbFrameLocated.size() << " images, located " << count << " images." << endl;

        return ((float) count / (float) mlbFrameLocated.size());
    }

}