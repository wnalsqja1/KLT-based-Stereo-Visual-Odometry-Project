#pragma once
#include "kvl_pe.h"

namespace Converter{
    cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
    {
        cv::Mat cvMat(4,4,CV_64FC1);
        for(int i=0;i<4;i++)
            for(int j=0; j<4; j++)
                cvMat.at<double>(i,j)=m(i,j);

        return cvMat.clone();
    }

    cv::Mat toCVMat(const g2o::SE3Quat &SE3){
        Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigMat);
    }

    g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
    {
        Eigen::Matrix<double,3,3> R;
        R << cvT.at<double>(0,0), cvT.at<double>(0,1), cvT.at<double>(0,2),
            cvT.at<double>(1,0), cvT.at<double>(1,1), cvT.at<double>(1,2),
            cvT.at<double>(2,0), cvT.at<double>(2,1), cvT.at<double>(2,2);

        Eigen::Matrix<double,3,1> t(cvT.at<double>(0,3), cvT.at<double>(1,3), cvT.at<double>(2,3));

        return g2o::SE3Quat(R,t);
    }


    // R|t Mat 
    cv::Mat toCVMat(const Eigen::Matrix3d& r, const Eigen::Vector3d& t){
        Eigen::Matrix<double,4,4> m;
        m<< r(0,0), r(0,1), r(0,2), t[0],
            r(1,0), r(1,1), r(1,2), t[1],
            r(2,0), r(2,1), r(2,2), t[2],
            0.0f, 0.0f, 0.0f, 1.0f;

        return toCvMat(m);
    }

    cv::Mat toCVMat(const Eigen::Matrix3d& r){
        Eigen::Matrix<double,3,3> R;
        R<< r(0,0), r(0,1), r(0,2),
            r(1,0), r(1,1), r(1,2),
            r(2,0), r(2,1), r(2,2);

        cv::Mat cvMat(3,3,CV_64FC1);
        for(int i=0;i<3;i++)
            for(int j=0; j<3; j++)
                cvMat.at<double>(i,j)=R(i,j);

        return cvMat.clone();        
    }

    cv::Mat toCVMat(const Eigen::Vector3d& t){
        Eigen::Matrix<double,3,1> tmp;
        tmp<<t[0],t[1],t[2];
        
        cv::Mat cvMat(3,1,CV_64FC1);
        for(int i=0;i<3;i++)
            for(int j=0; j<1; j++)
                cvMat.at<double>(i,j)=tmp(i,j);

        return cvMat.clone();
    }

    cv::Mat toCVMat44to34(Mat& m){
        m.pop_back();
        return m;
    }

    cv::Mat toCVMatInverse(Mat& m){
        m=m.inv();
        return m;
    }

    cv::Mat toCVMat34to44(Mat& m){
        Mat m1 = Mat::eye(4, 4, CV_64FC1);
        m.copyTo(m1.rowRange(0,3).colRange(0,4));
        return m1;
    }

    vector<cv::Point2d> toPoint2d(vector<cv::Point2f> point){
        vector<cv::Point2d> point_;
        for(int i=0;i<point.size();i++){
            point_.push_back(cv::Point2d((double)point[i].x,(double)point[i].y));
        }
        return point_;
    }

    cv::Mat toCVMat(vector<cv::Point2f> points1){
        cv::Mat tmp=Mat::eye(points1.size(),2,CV_64FC1);
        for(int i=0;i<tmp.rows;i++){
            tmp.at<double>(i,0)=points1[i].x;
            tmp.at<double>(i,1)=points1[i].y;
        }
        return tmp;
    }

    cv::Mat toCVMat(vector<cv::Point3d> points1){
        cv::Mat tmp=Mat::eye(points1.size(),3,CV_64FC1);
        for(int i=0;i<tmp.rows;i++){
            tmp.at<double>(i,0)=points1[i].x;
            tmp.at<double>(i,1)=points1[i].y;
            tmp.at<double>(i,2)=points1[i].z;
        }
        return tmp;
    }    
}
