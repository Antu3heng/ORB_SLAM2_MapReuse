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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>

#include "Serializer.h"

namespace ORB_SLAM2_MapReuse
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        // vars that no need to be access through a mutex
        //
        ar & nNextId & mnId;
        ar & const_cast<long unsigned int &>(mnFrameId);
        ar & const_cast<double &>(mTimeStamp);
        // Grid vars
        ar & const_cast<int &>(mnGridCols);
        ar & const_cast<int &>(mnGridRows);
        ar & const_cast<float &>(mfGridElementWidthInv);
        ar & const_cast<float &>(mfGridElementHeightInv);
        // Tracking vars
        ar & mnTrackReferenceForFrame & mnFuseTargetForKF;
        // Local mapping vars
        ar & mnBALocalForKF & mnBAFixedForKF;
        // Keyframe database vars
        ar & mnLoopQuery & mnLoopWords & mLoopScore & mnRelocQuery & mnRelocWords & mRelocScore;
        // Loop closing vars
        ar & mTcwGBA & mTcwBefGBA & mnBAGlobalForKF;
        // Calibration vars
        ar & const_cast<float &>(fx) & const_cast<float &>(fy) & const_cast<float &>(cx) & const_cast<float &>(cy);
        ar & const_cast<float &>(invfx) & const_cast<float &>(invfy);
        ar & const_cast<float &>(mbf) & const_cast<float &>(mb) & const_cast<float &>(mThDepth);
        // Number of KeyPoints
        ar & const_cast<int &>(N);
        // KeyPoints, stereo coordinate and descriptors vars
        ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeys);
        ar & const_cast<std::vector<cv::KeyPoint> &>(mvKeysUn);
        ar & const_cast<std::vector<float> &>(mvuRight);
        ar & const_cast<std::vector<float> &>(mvDepth);
        ar & mDescriptors;
        // BoW vars
        ar & mBowVec & mFeatVec;
        // Pose relative to parent vars
        ar & mTcp;
        // Scale vars
        ar & const_cast<int &>(mnScaleLevels) & const_cast<float &>(mfScaleFactor) & const_cast<float &>(mfLogScaleFactor);
        ar & const_cast<std::vector<float> &>(mvScaleFactors) & const_cast<std::vector<float> &>(mvLevelSigma2) & const_cast<std::vector<float> &>(mvInvLevelSigma2);
        // Image bounds and calibration vars
        ar & const_cast<int &>(mnMinX) & const_cast<int &>(mnMinY) & const_cast<int &>(mnMaxX) & const_cast<int &>(mnMaxY) & const_cast<cv::Mat &>(mK);

        // vars that need to be access through a mutex
        // SE3 pose, camera center and stereo middle point
        {
            unique_lock<mutex> lock_pose(mMutexPose);
            ar & Tcw & Twc & Ow;
            ar & Cw;
        }
        // MapPoints associated to keypoints
        {
            unique_lock<mutex> lock_feature(mMutexFeatures);
            ar & mvpMapPoints;
        }

        {
            unique_lock<mutex> lock_connection(mMutexConnections);
            // Grid vars
            ar & mGrid & mConnectedKeyFrameWeights & mvpOrderedConnectedKeyFrames & mvOrderedWeights;
            // Spanning Tree and Loop Edges
            ar & mbFirstConnection & mpParent & mspChildrens & mspLoopEdges;
            // Bad flags
            ar & mbNotErase & mbToBeErased & mbBad & mHalfBaseline;
        }
    }

public:
    KeyFrame();
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);
    KeyFrame(std::vector<MapPoint*> vpMapPoints, std::vector<cv::KeyPoint> vKeys, std::vector<cv::KeyPoint> vKeysUn,
             int N, cv::Mat Descriptors, int nMinX, int nMinY, int nMaxX, int nMaxY, int nGridCols, int nGridRows, Map* pMap,
             float fGridElementWidthInv, float fGridElementHeightInv, std::vector< std::vector <std::vector<size_t> > > Grid,
             std::vector<float> vuRight, std::vector<float> vDepth, int nScaleLevels, float fScaleFactor, float fLogScaleFactor,
             std::vector<float> vScaleFactors, std::vector<float> vLevelSigma2, std::vector<float> vInvLevelSigma2);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Functions for loading map
    void SetMap(Map *pMap);
    void SetKeyFrameDatabase(KeyFrameDatabase *pKeyFrameDB);
    void SetORBVocabulary(ORBVocabulary *pORBVocabulary);

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }

    std::vector< std::vector <std::vector<size_t> > > GetGrid() const;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    mutable std::mutex mMutexPose;
    mutable std::mutex mMutexConnections;
    mutable std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
