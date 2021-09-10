/**
 * @file Serializer.h
 * @author Xinjiang Wang (wangxj83@sjtu.edu.cn)
 * @brief  This file contains serialize function for some datatypes that are not defined by boost
 * @version 0.1
 * @date 2021-03-17
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)

namespace boost
{
    namespace serialization
    {
        template<class Archive>
        void serialize(Archive &ar, DBoW2::BowVector &BowVec, const unsigned int version)
        {
            ar & boost::serialization::base_object<std::map<unsigned int, double>>(BowVec);
        }

        template<class Archive>
        void serialize(Archive &ar, DBoW2::FeatureVector &FeatureVec, const unsigned int version)
        {
            ar & boost::serialization::base_object<std::map<unsigned int, std::vector<unsigned int>>>(FeatureVec);
        }

        template<class Archive>
        void save(Archive &ar, const ::cv::Mat &m, const unsigned int version)
        {
            cv::Mat m_ = m;
            
            ar & m_.cols & m_.rows;

            size_t elemtype = m_.type(); 
            bool continuous = m_.isContinuous();
            ar & elemtype & continuous;
            
            if (continuous)
            {
                size_t data_size = m_.rows * m_.cols * m_.elemSize();
                ar & boost::serialization::make_array(m_.ptr(), data_size);
            }
            else
            {
                size_t rows_size = m_.cols * m_.elemSize();
                for (int i = 0; i < m_.rows; i++)
                {
                    ar & boost::serialization::make_array(m_.ptr(i), rows_size);
                }
            }
        }

        template<class Archive>
        void load(Archive &ar, ::cv::Mat &m, const unsigned int version)
        {
            int cols, rows;
            size_t elem_type;
            bool continuous;
            ar & cols & rows & elem_type & continuous;

            m.create(rows, cols, elem_type);

            if (continuous)
            {
                size_t data_size = m.rows * m.cols * m.elemSize();
                ar & boost::serialization::make_array(m.ptr(), data_size);
            }
            else
            {
                size_t rows_size = m.cols * m.elemSize();
                for (int i = 0; i < m.rows; i++)
                {
                    ar & boost::serialization::make_array(m.ptr(i), rows_size);
                }
            }            
        }

        template<class Archive>
        void serialize(Archive &ar, ::cv::KeyPoint &keypoint, const unsigned int version)
        {
            ar & keypoint.pt.x & keypoint.pt.y & keypoint.size & keypoint.angle;
            ar & keypoint.response & keypoint.octave & keypoint.class_id;
        }
    }
}



#endif // SERIALIZER_H