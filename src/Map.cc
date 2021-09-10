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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2_MapReuse
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(auto sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(auto sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

std::map<long unsigned int, KeyFrame*> Map::GetAllKeyFramesUseMap()
{
    unique_lock<mutex> lock(mMutexMap);
    std::map<long unsigned int, KeyFrame*> kf_in_map;
    for(auto &kf:mspKeyFrames)
        kf_in_map[kf->mnId] = kf;
    return kf_in_map;
}

std::map<long unsigned int, MapPoint*> Map::GetAllMapPointsUseMap()
{
    unique_lock<mutex> lock(mMutexMap);
    std::map<long unsigned int, MapPoint*> mp_in_map;
    for(auto &mp:mspMapPoints)
        mp_in_map[mp->mnId] = mp;
    return mp_in_map;
}

bool Map::SaveMap(const string &filename)
{
    std::cout << std::endl << "Saving Map to " << filename << std::endl;
    ofstream f;
    f.open(filename.c_str(), ios_base::out|ios_base::binary);
    if(!f.is_open())
    {
        std::cerr << std::endl << "File open occur error!" << std::endl;
        return false;
    }

    // Save MapPoints
    // Save the number of mappoints
    size_t nMapPoints = mspMapPoints.size();
    f.write((char*)&nMapPoints, sizeof(size_t));
    // Save the mappoints sequentially
    for(auto &mp:mspMapPoints)
    {
        if(mp->isBad())
        {
            long unsigned int id = ULONG_MAX;
            f.write((char*)&id, sizeof(long unsigned int));
            continue;
        }
        // Save ID
        f.write((char*)&mp->mnId, sizeof(long unsigned int));
        // Save world position
        cv::Mat mpPos = mp->GetWorldPos();
        for(int i = 0; i < 3; i++)
            f.write((char*)&mpPos.at<float>(i), sizeof(float));
        // Save the descriptor
        cv::Mat mp_descriptor = mp->GetDescriptor();
        f.write((char*)&mp_descriptor.rows, sizeof(int));
        f.write((char*)&mp_descriptor.cols, sizeof(int));
        for(int i = 0; i < mp_descriptor.rows; i++)
        {
            for(int j = 0; j < mp_descriptor.cols; j++)
                f.write((char*)&mp_descriptor.at<unsigned char>(i, j), sizeof(unsigned char));
        }
        // // Save the reference KeyFrame ID
        // long unsigned int id = mp->GetReferenceKeyFrame()->mnId;
        // f.write((char*)&id, sizeof(long unsigned int));
        // Save the scale invariance distances
        float fMinDistance = mp->GetMinDistanceInvariance() / 0.8f;
        float fMaxDistance = mp->GetMaxDistanceInvariance() / 1.2f;
        f.write((char*)&fMinDistance, sizeof(float));
        f.write((char*)&fMaxDistance, sizeof(float));
    }
    std::cout << std::endl << nMapPoints << " MapPoints has been saved successfully!" << std::endl;

    // Save KeyFrames
    // Save the number of keyframes
    size_t nKeyFrames = mspKeyFrames.size();
    f.write((char*)&nKeyFrames, sizeof(size_t));
    // Save the keyframes sequentially
    for(auto &kf:mspKeyFrames)
    {
        if(kf->isBad())
        {
            long unsigned int id = ULONG_MAX;
            f.write((char*)&id, sizeof(long unsigned int));
            continue;
        }
        // Save ID
        f.write((char*)&kf->mnId, sizeof(long unsigned int));
        // Save the pose of keyframe
        cv::Mat Tcw = kf->GetPose();
        for(int i = 0; i < Tcw.rows; i++)
            for(int j = 0; j < Tcw.cols; j++)
                f.write((char*)&Tcw.at<float>(i, j), sizeof(float));
        // Save the number and the (un)distorted keypoint in the keyframe
        f.write((char*)&kf->N, sizeof(int));
        f.write((char*)&kf->mDescriptors.cols, sizeof(int));
        vector<MapPoint*> vpMapPoints = kf->GetMapPointMatches();
        for(int i = 0; i < kf->N; i++)
        {
            f.write((char*)&kf->mvKeys[i].pt.x, sizeof(float));
            f.write((char*)&kf->mvKeys[i].pt.y, sizeof(float));
            f.write((char*)&kf->mvKeys[i].size, sizeof(float));
            f.write((char*)&kf->mvKeys[i].angle, sizeof(float));
            f.write((char*)&kf->mvKeys[i].response, sizeof(float));
            f.write((char*)&kf->mvKeys[i].octave, sizeof(int));
            f.write((char*)&kf->mvKeysUn[i].pt.x, sizeof(float));
            f.write((char*)&kf->mvKeysUn[i].pt.y, sizeof(float));
            f.write((char*)&kf->mvKeysUn[i].size, sizeof(float));
            f.write((char*)&kf->mvKeysUn[i].angle, sizeof(float));
            f.write((char*)&kf->mvKeysUn[i].response, sizeof(float));
            f.write((char*)&kf->mvKeysUn[i].octave, sizeof(int));

            // Save the mDescriptors
            for(int j = 0; j < kf->mDescriptors.cols; j++)
                f.write((char*)&kf->mDescriptors.at<unsigned char>(i, j), sizeof(unsigned char));

            // Save the idx and ID of MapPoint observed
            long unsigned int mpId;
            if(vpMapPoints[i] == NULL)
                mpId = ULONG_MAX;
            else
                mpId = vpMapPoints[i]->mnId;
            f.write((char*)&mpId, sizeof(long unsigned int));
        }
        // Save the Bow
        // Save the BowVector
        size_t BowVecSize = kf->mBowVec.size();
        f.write((char*)&BowVecSize, sizeof(size_t));
        for(auto &bv:kf->mBowVec)
        {
            unsigned int bvFirst = bv.first;
            double bvSecond = bv.second;
            f.write((char*)&bvFirst, sizeof(unsigned int));
            f.write((char*)&bvSecond, sizeof(double));
        }
        // Save the FeatureVector
        size_t FeatVecSize = kf->mFeatVec.size();
        f.write((char*)&FeatVecSize, sizeof(size_t));
        for(auto &fv:kf->mFeatVec)
        {
            unsigned int fvFirst = fv.first;
            std::vector<unsigned int> fvSecond = fv.second;
            size_t fvSecondSize = fvSecond.size();
            f.write((char*)&fvFirst, sizeof(unsigned int));
            f.write((char*)&fvSecondSize, sizeof(size_t));
            for(int i = 0; i < fvSecondSize; i++)
                f.write((char*)&fvSecond[i], sizeof(unsigned int));
        }
        // Save the Image bounds
        f.write((char*)&kf->mnMinX, sizeof(int));
        f.write((char*)&kf->mnMinY, sizeof(int));
        f.write((char*)&kf->mnMaxX, sizeof(int));
        f.write((char*)&kf->mnMaxY, sizeof(int));
        // Save the Grid for matching(Grid parameters and mGrid)
        f.write((char*)&kf->mnGridCols, sizeof(int));
        f.write((char*)&kf->mnGridRows, sizeof(int));
        f.write((char*)&kf->mfGridElementWidthInv, sizeof(float));
        f.write((char*)&kf->mfGridElementHeightInv, sizeof(float));
        std::vector< std::vector <std::vector<size_t> > > kfGrid = kf->GetGrid();
        for(int i = 0; i < kf->mnGridCols; i++)
            for(int j = 0; j < kf->mnGridRows; j++)
            {
                size_t Gridsize = kfGrid[i][j].size();
                f.write((char*)&Gridsize, sizeof(size_t));
                for(int k = 0; k < Gridsize; k++)
                    f.write((char*)&kfGrid[i][j][k], sizeof(size_t));
            }
        // Save the scale
        f.write((char*)&kf->mnScaleLevels, sizeof(int));
        f.write((char*)&kf->mfScaleFactor, sizeof(float));
        f.write((char*)&kf->mfLogScaleFactor, sizeof(float));
    }
    for(auto &kf:mspKeyFrames)
    {
        // Save the ID of connected keyframes and its weight
        // TODO: remove connection process and use mappoint match to recover the keyframe connection
        std::set<KeyFrame *> kfConnection = kf->GetConnectedKeyFrames();
        size_t nkfConnection = kfConnection.size();
        f.write((char*)&nkfConnection, sizeof(size_t));
        for(auto &kfc:kfConnection)
        {
            int weight = kf->GetWeight(kfc);
            f.write((char*)&kfc->mnId, sizeof(long unsigned int));
            f.write((char*)&weight, sizeof(int));
        }
        // Save the ID of parent
        KeyFrame *kfp = kf->GetParent();
        long unsigned int kfpId;
        if(kfp == NULL)
            kfpId = ULONG_MAX;
        else
            kfpId = kfp->mnId;
        f.write((char*)&kfpId, sizeof(long unsigned int));
    }
    std::cout << std::endl << nKeyFrames << " KeyFrames has been saved successfully!" << std::endl;

    f.close();
    return true;
}

bool Map::LoadMap(const string &filename)
{
    std::cout << std::endl << "Loading Map from " << filename << std::endl;
    ifstream f;
    f.open(filename.c_str(), ios_base::in|ios_base::binary);
    if(!f.is_open())
    {
        std::cerr << std::endl << "File open occur error!" << std::endl;
        return false;
    }

    // Load MapPoints
    // Load the number of mappoints
    size_t nMapPoints;
    f.read((char*)&nMapPoints, sizeof(size_t));
    // Load the mappoints sequentially
    for(size_t nmp = 0; nmp < nMapPoints; nmp++)
    {
        // Load the ID
        long unsigned int id;
        f.read((char*)&id, sizeof(long unsigned int));
        // Load world position
        cv::Mat Position(3, 1, CV_32F);
        for(int i = 0; i < 3; i++)
            f.read((char*)&Position.at<float>(i), sizeof(float));
        // Load the descriptor
        int descriptorRows, descriptorCols;
        f.read((char*)&descriptorRows, sizeof(int));
        f.read((char*)&descriptorCols, sizeof(int));
        cv::Mat mp_descriptor(descriptorRows, descriptorCols, CV_8U);
        for(int i = 0; i < descriptorRows; i++)
        {
            for(int j = 0; j < descriptorCols; j++)
                f.read((char*)&mp_descriptor.at<unsigned char>(i, j), sizeof(unsigned char));
        }
        // // Load the reference KeyFrame ID
        // long unsigned int id;
        // f.read((char*)&id, sizeof(long unsigned int));
        // Load the scale invariance distances
        float fMinDistance, fMaxDistance;
        f.read((char*)&fMinDistance, sizeof(float));
        f.read((char*)&fMaxDistance, sizeof(float));
        // Create the MapPoint and add it to Map
        MapPoint *mp = new MapPoint(Position, this, mp_descriptor, fMinDistance, fMaxDistance);
        mp->mnId = id;
        AddMapPoint(mp);
    }
    std::cout << std::endl << nMapPoints << " MapPoints has been loaded successfully!" << std::endl;

    // Load KeyFrames
    // Load the number of keyframes
    size_t nKeyFrames;
    f.read((char*)&nKeyFrames, sizeof(size_t));
    std::vector<KeyFrame*> orderKeyframes;
    orderKeyframes.resize(nKeyFrames);
    // Load the keyframes sequentially
    for(size_t nkf = 0; nkf < nKeyFrames; nkf++)
    {
        // Load the ID
        long unsigned int id;
        f.read((char*)&id, sizeof(long unsigned int));
        // Load the pose of keyframe
        cv::Mat Tcw(4, 4, CV_32F);
        for(int i = 0; i < Tcw.rows; i++)
            for(int j = 0; j < Tcw.cols; j++)
                f.read((char*)&Tcw.at<float>(i, j), sizeof(float));
        // Load the number and the (un)distorted keypoint in the keyframe
        int N, descriptorCols;
        std::vector<cv::KeyPoint> vKeys, vKeysUn;
        f.read((char*)&N, sizeof(int));
        f.read((char*)&descriptorCols, sizeof(int));
        vKeys.resize(N);
        vKeysUn.resize(N);
        cv::Mat Descriptors(N, descriptorCols, CV_8U);
        std::vector<MapPoint*> vpMapPoints;
        vpMapPoints.resize(N);
        std::map<long unsigned int, MapPoint*> mpMapPointsInMap = GetAllMapPointsUseMap();
        for(int i = 0; i < N; i++)
        {
            f.read((char*)&vKeys[i].pt.x, sizeof(float));
            f.read((char*)&vKeys[i].pt.y, sizeof(float));
            f.read((char*)&vKeys[i].size, sizeof(float));
            f.read((char*)&vKeys[i].angle, sizeof(float));
            f.read((char*)&vKeys[i].response, sizeof(float));
            f.read((char*)&vKeys[i].octave, sizeof(int));
            f.read((char*)&vKeysUn[i].pt.x, sizeof(float));
            f.read((char*)&vKeysUn[i].pt.y, sizeof(float));
            f.read((char*)&vKeysUn[i].size, sizeof(float));
            f.read((char*)&vKeysUn[i].angle, sizeof(float));
            f.read((char*)&vKeysUn[i].response, sizeof(float));
            f.read((char*)&vKeysUn[i].octave, sizeof(int));

            // Load the mDescriptors
            for(int j = 0; j < Descriptors.cols; j++)
                f.read((char*)&Descriptors.at<unsigned char>(i, j), sizeof(unsigned char));

            // Load the mappoint that keyframe matchs
            long unsigned int mpId;
            f.read((char*)&mpId, sizeof(long unsigned int));
            if(mpId == ULONG_MAX)
                vpMapPoints[i] = static_cast<MapPoint*>(NULL);
            else
                vpMapPoints[i] = mpMapPointsInMap[mpId];
        }
        // Load the Bow
        // Load the BowVector
        size_t BowVecSize;
        f.read((char*)&BowVecSize, sizeof(size_t));
        DBoW2::BowVector BowVec;
        for(int i = 0; i < BowVecSize; i++)
        {
            unsigned int bvFirst;
            double bvSecond;
            f.read((char*)&bvFirst, sizeof(unsigned int));
            f.read((char*)&bvSecond, sizeof(double));
            BowVec[bvFirst] = bvSecond;
        }
        // Load the FeatureVector
        size_t FeatVecSize;
        f.read((char*)&FeatVecSize, sizeof(size_t));
        DBoW2::FeatureVector FeatVec;
        for(int i = 0; i < FeatVecSize; i++)
        {
            unsigned int fvFirst;
            std::vector<unsigned int> fvSecond;
            size_t fvSecondSize;
            f.read((char*)&fvFirst, sizeof(unsigned int));
            f.read((char*)&fvSecondSize, sizeof(size_t));
            fvSecond.resize(fvSecondSize);
            for(int j = 0; j < fvSecondSize; j++)
                f.read((char*)&fvSecond[j], sizeof(unsigned int));
            FeatVec[fvFirst] = fvSecond;
        }
        // Load the Image bound
        int nMinX, nMinY, nMaxX, nMaxY;
        f.read((char*)&nMinX, sizeof(int));
        f.read((char*)&nMinY, sizeof(int));
        f.read((char*)&nMaxX, sizeof(int));
        f.read((char*)&nMaxY, sizeof(int));
        // Load the Grid for speed matching(Grid parameters and mGrid)
        int nGridCols, nGridRows;
        float fGridElementWidthInv, fGridElementHeightInv;
        f.read((char*)&nGridCols, sizeof(int));
        f.read((char*)&nGridRows, sizeof(int));
        f.read((char*)&fGridElementWidthInv, sizeof(float));
        f.read((char*)&fGridElementHeightInv, sizeof(float));
        std::vector< std::vector <std::vector<size_t> > > Grid;
        Grid.resize(nGridCols);
        for(int i = 0; i < nGridCols; i++)
        {
            Grid[i].resize(nGridCols);
            for(int j = 0; j < nGridRows; j++)
            {
                size_t Gridsize;
                f.read((char*)&Gridsize, sizeof(size_t));
                Grid[i][j].resize(Gridsize);
                for(int k = 0; k < Gridsize; k++)
                    f.read((char*)&Grid[i][j][k], sizeof(size_t));
            }
        }
        // Load the scale
        int nScaleLevels;
        float fScaleFactor, fLogScaleFactor;
        f.read((char*)&nScaleLevels, sizeof(int));
        f.read((char*)&fScaleFactor, sizeof(float));
        f.read((char*)&fLogScaleFactor, sizeof(float));
        std::vector<float> vScaleFactors, vLevelSigma2, vInvLevelSigma2;
        vScaleFactors.resize(nScaleLevels);
        vLevelSigma2.resize(nScaleLevels);
        vInvLevelSigma2.resize(nScaleLevels);
        vScaleFactors[0] = 1.0f;
        vLevelSigma2[0]  = 1.0f;
        vInvLevelSigma2[0] =  1.0f / vLevelSigma2[0];
        for(int i = 1; i < nScaleLevels; i++)
        {
            vScaleFactors[i] = vScaleFactors[i - 1] * fScaleFactor;
            vLevelSigma2[i] = vScaleFactors[i] * vScaleFactors[i];
            vInvLevelSigma2[i]  = 1.0f/vLevelSigma2[i];
        }
        // Create the keyframe object and add it to the map
        // TODO: save and load the vuRight and vDepth
        std::vector<float> vuRight = vector<float>(N,-1);
        std::vector<float> vDepth = vector<float>(N,-1);
        KeyFrame *kf = new KeyFrame(vpMapPoints, vKeys, vKeysUn, N, Descriptors, nMinX, nMinY, nMaxX, nMaxY, nGridCols,
                                    nGridRows, this, fGridElementWidthInv, fGridElementHeightInv, Grid, vuRight, vDepth, nScaleLevels, fScaleFactor,
                                    fLogScaleFactor, vScaleFactors, vLevelSigma2, vInvLevelSigma2);
        kf->mnId = id;
        kf->SetPose(Tcw);
        kf->mBowVec = BowVec;
        kf->mFeatVec = FeatVec;
        std::vector<MapPoint*> kf_MapPointMatches = kf->GetMapPointMatches();
        for(size_t i = 0; i < N; i++)
        {
            if(kf_MapPointMatches[i])
            {
                kf_MapPointMatches[i]->AddObservation(kf, i);
                kf_MapPointMatches[i]->nObs++;
                if(!kf_MapPointMatches[i]->GetReferenceKeyFrame())
                    kf_MapPointMatches[i]->SetReferenceKeyFrame(kf);
            }
        }
        AddKeyFrame(kf);
        orderKeyframes[nkf] = kf;
    }
    std::map<long unsigned int, KeyFrame*> mpKeyFramesInMap = GetAllKeyFramesUseMap();
    for(size_t nkf = 0; nkf < nKeyFrames; nkf++)
    {
        // Load the ID of connected keyframes and its weight
        size_t nkfConnection;
        f.read((char*)&nkfConnection, sizeof(size_t));
        for(int i = 0; i < nkfConnection; i++)
        {
            long unsigned int kfId;
            int weight;
            f.read((char*)&kfId, sizeof(long unsigned int));
            f.read((char*)&weight, sizeof(int));
            orderKeyframes[nkf]->AddConnection(mpKeyFramesInMap[kfId], weight);
        }
        // Load the ID of parent
        long unsigned int kfpId;
        f.read((char*)&kfpId, sizeof(long unsigned int));
        if(kfpId != ULONG_MAX)
            orderKeyframes[nkf]->ChangeParent(mpKeyFramesInMap[kfpId]);
    }
    std::cout << std::endl << nKeyFrames << " KeyFrames has been loaded successfully!" << std::endl;

    f.close();
    return true;
}

} //namespace ORB_SLAM
