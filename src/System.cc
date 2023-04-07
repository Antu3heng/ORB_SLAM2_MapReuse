/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2_MapReuse
{

    System::System(const string &strSettingsFile, const eSensor sensor, const eMode mode, const bool bUseViewer)
            : mSensor(sensor), mMode(mode), mpViewer(static_cast<Viewer *>(nullptr)), mbReset(false),
              mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false)
    {
        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        string strVocFile;
        fsSettings["Vocabulary.Path"] >> strVocFile;

        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create the Map
        mpMap = new Map();

        image_height = fsSettings["Camera.height"];
        image_width = fsSettings["Camera.width"];

        //Create the main process thread(tracker or locator)
        if (mMode != VisualLocalization)
        {
            //Create KeyFrame Database
            mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

            cout << "Input sensor was set to: ";

            if (mSensor == MONOCULAR)
                cout << "Monocular" << endl;
            else if (mSensor == STEREO)
                cout << "Stereo" << endl;
            else if (mSensor == RGBD)
                cout << "RGB-D" << endl;

            if (mMode == Localization)
            {
                mbActivateLocalizationMode = true;
                cout << endl << "Localization with ORB-SLAM2 map...." << endl;
            } else
                cout << endl << "Running ORB-SLAM2 ...." << endl;

            //Create Drawers. These are used by the Viewer
            mpFrameDrawer = new FrameDrawer(mpMap);
            mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

            //Initialize the Tracking thread
            //(it will live in the main thread of execution, the one that called this constructor)
            mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                     mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

            //Initialize the Local Mapping thread and launch
            mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
            mptLocalMapping = new thread(&ORB_SLAM2_MapReuse::LocalMapping::Run, mpLocalMapper);

            //Initialize the Loop Closing thread and launch
            mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
            mptLoopClosing = new thread(&ORB_SLAM2_MapReuse::LoopClosing::Run, mpLoopCloser);

            //Initialize the Viewer thread and launch
            if (bUseViewer)
            {
                mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
                mptViewer = new thread(&Viewer::Run, mpViewer);
                mpTracker->SetViewer(mpViewer);
            }

            //Set pointers between threads
            mpTracker->SetLocalMapper(mpLocalMapper);
            mpTracker->SetLoopClosing(mpLoopCloser);

            mpLocalMapper->SetTracker(mpTracker);
            mpLocalMapper->SetLoopCloser(mpLoopCloser);

            mpLoopCloser->SetTracker(mpTracker);
            mpLoopCloser->SetLocalMapper(mpLocalMapper);
        } else
        {
            int nAblation = fsSettings["Localization.AblationTest"];
            bAblation = nAblation;
            if (bAblation)
                cout << "Running ablation study!!!" << endl;
            int nMatchingUsingBooster = fsSettings["Localization.MatchingUsingBooster"];
            bMatchingUsingBooster = nMatchingUsingBooster;
            if (bMatchingUsingBooster)
                cout << "Feature Matching using ORB+Boost-B!!!" << endl;
            int nRetrievalUsingBooster = fsSettings["Localization.RetrievalUsingBooster"];
            bRetrievalUsingBooster = nRetrievalUsingBooster;
            if (bRetrievalUsingBooster)
                cout << "Image Retrieval using ORB+Boost-B!!!" << endl;

            string strVocFileForBooster;
            fsSettings["Localization.BoosterVocPath"] >> strVocFileForBooster;

            //Load ORB Vocabulary
            cout << endl << "Loading ORB Vocabulary for ORB+Boost-B. This could take a while..." << endl;

            mpVocabularyForBoostingAblation = new ORBVocabulary();
            bool bLoad = mpVocabularyForBoostingAblation->loadFromTextFile(strVocFileForBooster);
            if (!bLoad)
            {
                cerr << "Wrong path to vocabulary. " << endl;
                cerr << "Failed to open at: " << strVocFileForBooster << endl;
                exit(-1);
            }
            cout << "Vocabulary for ORB+Boost-B loaded!" << endl << endl;

            //Create KeyFrame Database
            if (bRetrievalUsingBooster)
                mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabularyForBoostingAblation);
            else
                mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

            cout << endl << "Visual Localization with ORB-SLAM2 map...." << endl;
            mpLocator = new Locator(mpVocabulary, mpKeyFrameDatabase, strSettingsFile, bAblation, bMatchingUsingBooster, bRetrievalUsingBooster, mpVocabularyForBoostingAblation);
        }

    }

    cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
    {
        if (mSensor != STEREO)
        {
            cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depth, const double &timestamp)
    {
        if (mSensor != RGBD)
        {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depth, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        if (mSensor != MONOCULAR)
        {
            cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

        return Tcw;
    }

    cv::Mat System::ORBLocalization(const cv::Mat &im, const double &timestamp)
    {
        return mpLocator->visualLocalization(im, timestamp);
    }

    void System::ActivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    void System::DeactivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    bool System::MapChanged()
    {
        static int n = 0;
        int curn = mpMap->GetLastBigChangeIdx();
        if (n < curn)
        {
            n = curn;
            return true;
        } else
            return false;
    }

    void System::Reset()
    {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    void System::Shutdown()
    {
        if (mMode != VisualLocalization)
        {
            mpLocalMapper->RequestFinish();
            mpLoopCloser->RequestFinish();
            if (mpViewer)
            {
                mpViewer->RequestFinish();
                while (!mpViewer->isFinished())
                    usleep(5000);
            }

            // Wait until all thread have effectively stopped
            while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
            {
                usleep(5000);
            }

            if (mpViewer)
                pangolin::BindToContext("ORB-SLAM2: Map Viewer");
        }
    }

    void System::SaveTrajectoryTUM(const string &filename)
    {
        cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        if (mMode != VisualLocalization)
        {
            vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
            sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

            // Transform all keyframes so that the first keyframe is at the origin.
            // After a loop closure the first keyframe might not be at the origin.
            cv::Mat Two = vpKFs[0]->GetPoseInverse();

            // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
            // We need to get first the keyframe pose and then concatenate the relative transformation.
            // Frames not localized (tracking failure) are not saved.

            // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
            // which is true when tracking failed (lbL).
            auto lRit = mpTracker->mlpReferences.begin();
            auto lT = mpTracker->mlFrameTimes.begin();
            auto lbL = mpTracker->mlbLost.begin();
            for (auto lit = mpTracker->mlRelativeFramePoses.begin(),
                         lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
            {
                if (*lbL)
                    continue;

                KeyFrame *pKF = *lRit;

                cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

                // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
                while (pKF->isBad())
                {
                    Trw = Trw * pKF->mTcp;
                    pKF = pKF->GetParent();
                }

                Trw = Trw * pKF->GetPose() * Two;

                cv::Mat Tcw = (*lit) * Trw;
                cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

                vector<float> q = Converter::toQuaternion(Rwc);

                f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1)
                  << " "
                  << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
            }
        } else
        {
            auto itT = mpLocator->mlFrameTimes.begin();
            auto itFlag = mpLocator->mlbFrameLocated.begin();
            for (auto it = mpLocator->mlFramePoses.begin(), itEnd = mpLocator->mlFramePoses.end();
                 it != itEnd; ++it, ++itT, ++itFlag)
            {
                if (!(*itFlag))
                {
                    // f << setprecision(6) << *itT << " " << setprecision(9) << double(0) << " " << double(0) << " "
                    //   << double(0) << " " << double(0) << " " << double(0) << " " << double(0) << " " << double(0)
                    //   << endl;
                    continue;
                }

                cv::Mat Rwc = (*it).rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * (*it).rowRange(0, 3).col(3);
                vector<float> q = Converter::toQuaternion(Rwc);

                f << setprecision(6) << *itT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1)
                  << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
            }

            float recallRate = mpLocator->CalculateRecall();;
            cout << "Recall rate: " << recallRate << endl;
        }

        f.close();
        cout << endl << "TUM format trajectory saved!" << endl;
    }


    void System::SaveKeyFrameTrajectoryTUM(const string &filename)
    {
        if (mMode != SLAM)
        {
            cout << endl << "Saving keyframe pose to " << filename << " ..." << endl;
        }
        else
            cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        //cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];

            // pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1)
              << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }

        f.close();
        cout << endl << "TUM format keyframe trajectory saved!" << endl;
    }

    void System::SaveTrajectoryKITTI(const string &filename)
    {
        cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

        ofstream f;
        f.open(filename.c_str());
        f << fixed;


        if (mMode != VisualLocalization)
        {
            vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
            sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

            // Transform all keyframes so that the first keyframe is at the origin.
            // After a loop closure the first keyframe might not be at the origin.
            cv::Mat Two = vpKFs[0]->GetPoseInverse();

            // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
            // We need to get first the keyframe pose and then concatenate the relative transformation.
            // Frames not localized (tracking failure) are not saved.

            // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
            // which is true when tracking failed (lbL).
            auto lRit = mpTracker->mlpReferences.begin();
            auto lT = mpTracker->mlFrameTimes.begin();
            for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
                 lit != lend; lit++, lRit++, lT++)
            {
                ORB_SLAM2_MapReuse::KeyFrame *pKF = *lRit;

                cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

                while (pKF->isBad())
                {
                    //  cout << "bad parent" << endl;
                    Trw = Trw * pKF->mTcp;
                    pKF = pKF->GetParent();
                }

                Trw = Trw * pKF->GetPose() * Two;

                cv::Mat Tcw = (*lit) * Trw;
                cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

                f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2)
                  << " " << twc.at<float>(0) << " " <<
                  Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
                  << twc.at<float>(1)
                  << " " <<
                  Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
                  << twc.at<float>(2)
                  << endl;
            }
        } else
        {
            auto itFlag = mpLocator->mlbFrameLocated.begin();
            for (auto it = mpLocator->mlFramePoses.begin(), itEnd = mpLocator->mlFramePoses.end();
                 it != itEnd; ++it, ++itFlag)
            {
                if (!(*itFlag))
                    continue;

                cv::Mat Rwc = (*it).rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * (*it).rowRange(0, 3).col(3);

                f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2)
                  << " " << twc.at<float>(0) << " " <<
                  Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
                  << twc.at<float>(1) << " " <<
                  Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
                  << twc.at<float>(2) << endl;
            }

            mpLocator->CalculateRecall();
        }

        f.close();
        cout << endl << "KITTI format trajectory saved!" << endl;
    }

    void System::SaveKeyFrameTrajectoryKITTI(const string &filename)
    {
         if (mMode != SLAM)
        {
            cout << endl << "Saving keyframe pose to " << filename << " ..." << endl;
        }
        else
            cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];

            pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(9) << R.at<float>(0, 0) << " " << R.at<float>(0, 1) << " " << R.at<float>(0, 2)
                  << " " << t.at<float>(0) << " " <<
                  R.at<float>(1, 0) << " " << R.at<float>(1, 1) << " " << R.at<float>(1, 2) << " "
                  << t.at<float>(1)
                  << " " <<
                  R.at<float>(2, 0) << " " << R.at<float>(2, 1) << " " << R.at<float>(2, 2) << " "
                  << t.at<float>(2)
                  << endl;

        }

        f.close();
        cout << endl << "KITTI format keyframe trajectory saved!" << endl;
    }

    void System::SaveRelocCandidateKF(const string &filename)
    {
        cout << endl << "Saving relocalization correspondence to " << filename << " ..." << endl;

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        f << "Query image timestamp    Candidate keyframe timestamp" << endl;

        auto itlCandKF = mpLocator->mlCandidateKFTimes.begin();
        for (auto lit = mpLocator->mlFrameTimes.begin(), lend = mpLocator->mlFrameTimes.end();
             lit != lend; ++lit, ++itlCandKF)
        {
            f << setprecision(9) << *lit;
            std::vector<double> vCandKFTime = *itlCandKF;
            for (auto &CKF: vCandKFTime)
                f << " " << setprecision(9) << CKF;
            f << endl;
        }

        f.close();
        cout << endl << "localization correspondence saved!" << endl;
    }

    void System::SaveRelocKF(const string &filename)
    {
        cout << endl << "Saving relocalization keyframe used to " << filename << " ..." << endl;

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        f << "Query image timestamp    Keyframe used to relocate timestamp" << endl;

        auto itlRelocKF = mpLocator->mlRelocKFTimes.begin();
        for (auto lit = mpLocator->mlFrameTimes.begin(), lend = mpLocator->mlFrameTimes.end();
             lit != lend; ++lit, ++itlRelocKF)
        {
            f << setprecision(9) << *lit << " " << *itlRelocKF << endl;
        }

        f.close();
        cout << endl << "localization keyframe saved!" << endl;
    }

    int System::GetTrackingState()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackingState;
    }

    vector<MapPoint *> System::GetTrackedMapPoints()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedMapPoints;
    }

    vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedKeyPointsUn;
    }

    bool System::SaveMap(const string &filename)
    {
        cout << endl << "Map save to " << filename << endl;

        if (!mpMap->SaveMap(filename))
            return false;

        cout << endl << "Map has been saved successfully!" << endl;
        return true;
    }

    bool System::LoadMap(const string &filename)
    {
        cout << endl << "Map load from " << filename << endl;

        if (!mpMap->LoadMap(filename))
            return false;

        if (bAblation)
        {
            std::vector<KeyFrame *> vpKeyFrames = mpMap->GetAllKeyFrames();
            for (auto &kf: vpKeyFrames)
            {
                kf->SetKeyFrameDatabase(mpKeyFrameDatabase);
                kf->SetORBVocabulary(mpVocabulary);
                if (bMatchingUsingBooster && !bRetrievalUsingBooster)
                {
                    kf->mFeatVec.clear();
                    // Boosting the ORB
                    mpLocator->mORBrefiner->refine(image_height, image_width, kf->mvKeys, kf->mDescriptors);
                    // Update the FeatVec for matching using voc for ORB+Boost-B
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(kf->mDescriptors);
                    DBoW2::BowVector fakeBowVec;
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,fakeBowVec,kf->mFeatVec,4);
                }
                else if (!bMatchingUsingBooster && bRetrievalUsingBooster)
                {
                    kf->mBowVec.clear();
                    // Creat a container for ORB+Boost-B
                    cv::Mat Descriptors = kf->mDescriptors;
                    // Boosting the ORB using the container
                    mpLocator->mORBrefiner->refine(image_height, image_width, kf->mvKeys, Descriptors);
                    // Update the BowVec for retrieval using voc for ORB+Boost-B
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(Descriptors);
                    DBoW2::FeatureVector fakeFeatVec;
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,kf->mBowVec,fakeFeatVec,4);
                }
                else if (bMatchingUsingBooster && bRetrievalUsingBooster)
                {
                    kf->mFeatVec.clear();
                    kf->mBowVec.clear();
                    // Boosting the ORB
                    mpLocator->mORBrefiner->refine(image_height, image_width, kf->mvKeys, kf->mDescriptors);
                    // compute the Bow
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(kf->mDescriptors);
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,kf->mBowVec,kf->mFeatVec,4);
                }
                mpKeyFrameDatabase->add(kf);
            }
            std::vector<MapPoint*> vpMapPoints = mpMap->GetAllMapPoints();
            for(auto &mp:vpMapPoints)
                if(mp)
                    mp->ComputeDistinctiveDescriptors();

            cout << endl << "Map has been loaded and converted successfully!" << endl;
            return true;
        }
        else
        {
            // std::vector<MapPoint*> vpMapPoints = mpMap->GetAllMapPoints();
            // for(auto &mp:vpMapPoints)
            //     if(mp)
            //     {
            //         mp->ComputeDistinctiveDescriptors();
            //         // TODO: need to recover the ScaleFactors of keyframe
            //         // mp->UpdateNormalAndDepth();
            //     }
            std::vector<KeyFrame *> vpKeyFrames = mpMap->GetAllKeyFrames();
            for (auto &kf: vpKeyFrames)
            {
                kf->SetKeyFrameDatabase(mpKeyFrameDatabase);
                kf->SetORBVocabulary(mpVocabulary);
                // kf->ComputeBoW();
                // kf->UpdateConnections();
                mpKeyFrameDatabase->add(kf);
            }
            cout << endl << "Map has been loaded successfully!" << endl;
            return true;
        }
    }

    bool System::SaveMapUsingBoost(const std::string &filename)
    {
        cout << endl << "Map save to " << filename << endl;

        std::ofstream os(filename);

        boost::archive::binary_oarchive oa(os, boost::archive::no_header);
        oa << mpMap;

        cout << endl << "Map has been saved successfully!" << endl;
        return true;
    }

    bool System::LoadMapUsingBoost(const std::string &filename)
    {
        cout << endl << "Map load from " << filename << endl;

        std::ifstream is(filename);

        Map *temp = mpMap;
        boost::archive::binary_iarchive ia(is, boost::archive::no_header);
        ia >> mpMap;

        /**
         * @brief TODO
         * 1. Spanning Tree may have some problem
         * 2. SLAM Mode too many KeyFrames
         */
        std::vector<KeyFrame *> vpKeyFrames = mpMap->GetAllKeyFrames();
        for (auto &vpKeyFrame: vpKeyFrames)
        {
            vpKeyFrame->SetKeyFrameDatabase(mpKeyFrameDatabase);
            vpKeyFrame->SetORBVocabulary(mpVocabulary);
            vpKeyFrame->SetMap(mpMap);
            mpKeyFrameDatabase->add(vpKeyFrame);
            vpKeyFrame->UpdateConnections();

            if (bAblation)
            {
                vpKeyFrame->SetKeyFrameDatabase(mpKeyFrameDatabase);
                vpKeyFrame->SetORBVocabulary(mpVocabulary);
                if (bMatchingUsingBooster && !bRetrievalUsingBooster)
                {
                    vpKeyFrame->mFeatVec.clear();
                    // Boosting the ORB
                    mpLocator->mORBrefiner->refine(image_height, image_width, vpKeyFrame->mvKeys, vpKeyFrame->mDescriptors);
                    // Update the FeatVec for matching using voc for ORB+Boost-B
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(vpKeyFrame->mDescriptors);
                    DBoW2::BowVector fakeBowVec;
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,fakeBowVec,vpKeyFrame->mFeatVec,4);
                }
                else if (!bMatchingUsingBooster && bRetrievalUsingBooster)
                {
                    vpKeyFrame->mBowVec.clear();
                    // Creat a container for ORB+Boost-B
                    cv::Mat Descriptors = vpKeyFrame->mDescriptors;
                    // Boosting the ORB using the container
                    mpLocator->mORBrefiner->refine(image_height, image_width, vpKeyFrame->mvKeys, Descriptors);
                    // Update the BowVec for retrieval using voc for ORB+Boost-B
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(Descriptors);
                    DBoW2::FeatureVector fakeFeatVec;
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,vpKeyFrame->mBowVec,fakeFeatVec,4);
                }
                else if (bMatchingUsingBooster && bRetrievalUsingBooster)
                {
                    vpKeyFrame->mFeatVec.clear();
                    vpKeyFrame->mBowVec.clear();
                    // Boosting the ORB
                    mpLocator->mORBrefiner->refine(image_height, image_width, vpKeyFrame->mvKeys, vpKeyFrame->mDescriptors);
                    // compute the Bow
                    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(vpKeyFrame->mDescriptors);
                    mpVocabularyForBoostingAblation->transform(vCurrentDesc,vpKeyFrame->mBowVec,vpKeyFrame->mFeatVec,4);
                }
            }
        }

        std::vector<MapPoint *> vpMapPoints = mpMap->GetAllMapPoints();
        for (auto &vpMapPoint: vpMapPoints)
        {
            vpMapPoint->SetMap(mpMap);
            if (bAblation)
                vpMapPoint->ComputeDistinctiveDescriptors();
        }

        if (mMode != VisualLocalization)
        {
            mpFrameDrawer->SetMap(mpMap);
            mpMapDrawer->mpMap = mpMap;
            mpTracker->SetMap(mpMap);
            mpLocalMapper->SetMap(mpMap);
            mpLoopCloser->SetMap(mpMap);
        }

        delete temp;
        temp = nullptr;

        cout << endl << "Map loaded successfully." << endl;
        return true;
    }

    void System::print_features_nums()
    {
        cout << endl << "avg feat num: " << (float)mpTracker->total_num_features / (float)mpTracker->count << endl;
    }

} //namespace ORB_SLAM
