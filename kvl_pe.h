#pragma once
//opencv library
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core.hpp"

#include <iostream>
#include <iterator>
#include <string>

//~'22.03.04.4
#include <map> // for arrign 3d world point - 2d feature point

//~'22.03.21.


//~'22.03.28.
#include <cmath>
#include <random>
#include <utility>//using pair, vector

//~'22.04.05.
#include <cstring>
#include <fstream>

//~'22.04.12.
#include<opencv/cv.h>

//~'22.07.26.
#include "Thirdparty/DBoW2/include/DBoW2/DBoW2.h"

#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"

#include "Thirdparty/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/eigen/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/csparse/linear_solver_csparse.h"
#include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"

#include "Thirdparty/g2o/g2o/types/slam3d/vertex_se3.h"
#include "Thirdparty/g2o/g2o/types/slam3d/edge_se3.h"
#include <Thirdparty/g2o/g2o/types/slam3d/types_slam3d.h>
#include <Thirdparty/g2o/g2o/types/slam3d/se3quat.h>

#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/sim3/types_seven_dof_expmap.h"
#include <vector>

using namespace cv;
using namespace std;
using namespace DBoW2;

#define MAX_FRAME 4540
#define MIN_NUM_FEAT 2000

bool outlier_over = false;
bool redetection_switch = false;
int mode = 0;

double focal = 718.8560;
cv::Point2d pp(607.1928, 185.2157);

double m[] = { 718.8560, 0.f, 607.1928, 0.f, 718.8560, 185.2157, 0.0f, 0.0f, 1.0f };	// left intrinsic parameters
Mat A(3, 3, CV_64FC1, m);	// camera matrix

double m1[] = { 718.8560, 0.f, 607.1928, -386.1448, 0.0f, 718.8560, 185.2157, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
Mat A1(3, 4, CV_64FC1, m1); //right intrinsic parameters


/* for loop closing */

#define MAX_IMAGE_NUMBER 5000


double threshold_score=0.16;


void featureDetection(Mat img_1, vector<Point2f>& points1) {   //uses FAST as of now, modify parameters as necessary
    // goodFeaturesToTrack(img_1, points1, 1000, 0.01, 20);
    goodFeaturesToTrack(img_1, points1, 1500, 0.01, 15);

}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, vector<Point2f>& keyFeatures_removed) {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            keyFeatures_removed.erase(keyFeatures_removed.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status) {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, vector<Point2f>& keyFeatures_removed,  vector<int>& index_of_landmark) {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            index_of_landmark.erase(index_of_landmark.begin()+(i-indexCorrection));
            keyFeatures_removed.erase(keyFeatures_removed.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

Mat featureTracking_3(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, Mat point3d) {

    //this function automatically gets rid of points for which tracking fails
    //printf("point size %d, point3d size : %d\n", points1.size(), point3d.rows);
    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //printf("points1 ���� %d, points2 ���� %d\n", points1.size(), points2.size());
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;

    //printf("start featrue point size rows :%d  cols : %d", point3d.rows, point3d.cols);
    vector<cv::Point3f> point3d_homo;
    for (int i = 0; i < point3d.rows; i++) {
        point3d_homo.push_back
        (cv::Point3f(point3d.at<double>(i, 0), point3d.at<double>(i, 1), point3d.at<double>(i, 2)));
    }

    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            point3d_homo.erase(point3d_homo.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }

    Mat tmpMat = Mat::eye(points1.size(), 3, CV_64FC1);
    for (int i = 0; i < points1.size(); i++) {
        tmpMat.at<double>(i, 0) = (double)point3d_homo[i].x;
        tmpMat.at<double>(i, 1) = (double)point3d_homo[i].y;
        tmpMat.at<double>(i, 2) = (double)point3d_homo[i].z;
    }
    return tmpMat;
}

//after keyframenum 0  
Mat featureTracking_3(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, Mat& point3d, vector<int>& index_of_landmark) {

    //this function automatically gets rid of points for which tracking fails
    //printf("point size %d, point3d size : %d\n", points1.size(), point3d.rows);
    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 40, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //printf("points1 ���� %d, points2 ���� %d\n", points1.size(), points2.size());
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;

    //printf("start featrue point size rows :%d  cols : %d", point3d.rows, point3d.cols);
    vector<cv::Point3f> point3d_homo;
    for (int i = 0; i < point3d.rows; i++) {
        point3d_homo.push_back
        (cv::Point3f(point3d.at<double>(i, 0), point3d.at<double>(i, 1), point3d.at<double>(i, 2)));
    }

    for (int i = 0; i < status.size(); i++)
    {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            point3d_homo.erase(point3d_homo.begin() + (i - indexCorrection));
            index_of_landmark.erase(index_of_landmark.begin()+(i-indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }

    Mat tmpMat = Mat::eye(points1.size(), 3, CV_64FC1);
    for (int i = 0; i < points1.size(); i++) {
        tmpMat.at<double>(i, 0) = (double)point3d_homo[i].x;
        tmpMat.at<double>(i, 1) = (double)point3d_homo[i].y;
        tmpMat.at<double>(i, 2) = (double)point3d_homo[i].z;
    }
    point3d=tmpMat.clone();
    return tmpMat;
}

Mat convert_14to13(Mat point3d_homo) {

    transpose(point3d_homo, point3d_homo);
    for (int i = 0; i < point3d_homo.rows; i++) {
        double temp = point3d_homo.at<double>(i, 3);
        for (int j = 0; j < 4; j++)
        {
            point3d_homo.at<double>(i, j) = point3d_homo.at<double>(i, j) / temp;
        }
    }

    point3d_homo.convertTo(point3d_homo, CV_64FC1);
    Mat tmpMat= point3d_homo.rowRange(0, point3d_homo.rows).colRange(0, 3).clone();
    return tmpMat;
}



/*
    UpdateRT functions :: Update Rotation, Translation Matrix Using outputs(rvec,tvec) of SolvePnP or RecoverPose
    Store_inlier functions :: inlier of 3D map points, 2D features 
*/

void UpdateRT(Mat& Rt0, Mat& Rt1, Mat& rvec, Mat& tvec, int mode){
    Rt0 = Mat::eye(3, 4, CV_64FC1); //prev pose
    Rt0 = Rt1.clone();
    Rt1 = Mat::eye(3, 4, CV_64FC1); //next pose

    if(mode==1){
        Mat R1;
        Rodrigues(rvec, R1);

        R1.copyTo(Rt1.rowRange(0, 3).colRange(0, 3));
        tvec.copyTo(Rt1.rowRange(0, 3).col(3));        
    }
    else if(mode==0){
        rvec.copyTo(Rt1.rowRange(0, 3).colRange(0, 3));
        tvec.copyTo(Rt1.rowRange(0, 3).col(3));
    }
    Rt0=Rt1.clone();
}


void Store_inlier(Mat inlier, vector<Point2f>& currFeatures, Mat& point3d_world, 
                    vector<Point2d>& tempFeatures, vector<Point3d>& tempFeatures1,
                    vector<int>& index_of_landmark, vector<int>& index_of_landmark_inlier){

    vector<int> dst1;
    for (int i = 0; i < inlier.rows; i++) {
        dst1.push_back(inlier.at<int>(i, 0));
    }

    for(int i=0;i<point3d_world.rows;i++){

        if(i==dst1.front()){
            dst1.erase(dst1.begin());
            index_of_landmark_inlier.push_back(index_of_landmark[i]);
            tempFeatures.push_back(cv::Point2f((double)currFeatures[i].x, (double)currFeatures[i].y));
            tempFeatures1.push_back(cv::Point3f(point3d_world.at<double>(i,0), point3d_world.at<double>(i,1),point3d_world.at<double>(i,2)));
        }

        if(dst1.empty()==true){
            break;
        }
    }
}

// Stereo Mode =========================================================================
void featureMatching(vector<Point2f>& points_L, vector<Point2f>& points_R, 
                            Mat& Rt0, Mat& Rt1, Mat& mask){

    double a1=(-386.1448);
    double a2=0.0f;
    double a3=0.0f;
    // findEssentialMat(points_R, points_L, focal, pp, RANSAC,0.999, 1.0, mask);
    /* Epipolar Constraints :: E=[t]_{x}R */ 
    double t[]={0.0f, -a3, a2,
                a3, 0.0f, -a1,
                -a2, a1, 0.0f};

    Mat skewt(3,3,CV_64FC1,t);
    Mat R=Mat::eye(3,3,CV_64FC1);
    Mat E=skewt*R;

    vector<Point2f> points_L_temp;
    vector<Point2f> points_R_temp;

    

    /* x1^T E  x0 = 0 :: dot( l0, x0 ) = 0 :: epipolar constraints */
    for(int i=0;i<points_L.size();i++){
        double xx1[]={points_L[i].x,points_L[i].y,1};
        Mat x1t(1,3,CV_64FC1,xx1);
        Mat tmpmat=x1t*E;
        double value=(tmpmat.at<double>(0,0)*points_R[i].x+
        tmpmat.at<double>(0,1)*points_R[i].y+
        tmpmat.at<double>(0,2)*1);

        float value1=abs(points_L[i].y-points_R[i].y);

        if(value1<1){
            points_L_temp.push_back(cv::Point2f(points_L[i].x,points_L[i].y));
            points_R_temp.push_back(cv::Point2f(points_R[i].x,points_R[i].y));            
        }
    }

    printf("hi, points size : %d\n",points_L_temp.size());
    points_L=points_L_temp;
    points_R=points_R_temp;

}

void Triangulation(vector<Point2f>& points1, vector<Point2f>& points2, Mat& Rt0, Mat& Rt1, Mat& point3d_world, Mat& Kd, Mat& Kd1, Mat& mask){
    vector<cv::Point2d> triangulation_points1, triangulation_points2;
    vector<cv::Point2f> triangulation_points1_float, triangulation_points2_float;

    Rt1=Mat::eye(4,4,CV_64FC1);
    Rt0.copyTo(Rt1.rowRange(0,3).colRange(0,4));
    Mat point3d_homo;
    for(int i=0;i<points1.size();i++){
        triangulation_points1.push_back
            (cv::Point2d((double)points1[i].x, (double)points1[i].y));
        triangulation_points2.push_back
            (cv::Point2d((double)points2[i].x, (double)points2[i].y));
        // triangulation_points1_float.push_back
        //     (cv::Point2f(points1[i].x, points1[i].y));
        // triangulation_points2_float.push_back
        //     (cv::Point2f(points2[i].x, points2[i].y));                
    }
    triangulatePoints(Kd * Rt0, Kd1 * Rt1, triangulation_points1, triangulation_points2, point3d_homo);
    point3d_world = convert_14to13(point3d_homo);
    
    // points1=triangulation_points1_float;
    // points2=triangulation_points2_float;
}

void VectorTypeCasting_FtoT(vector<Point2f>& points1, vector<Point2f>& points2, vector<Point2d>& temp_points1, vector<Point2d>& temp_points2){
    for(int i=0;i<points1.size();i++){
        temp_points1.push_back(
            cv::Point2d(double(points1[i].x),double(points1[i].y)));
        temp_points2.push_back(
            cv::Point2d(double(points2[i].x),double(points2[i].y)));        
    }
}

void VectorTypeCasting_FtoT(vector<Point2f>& points1,vector<Point2d>& temp_points1){
    for(int i=0;i<points1.size();i++){
        temp_points1.push_back(
            cv::Point2d(double(points1[i].x),double(points1[i].y)));
    }
}








vector<cv::Point2d> toPoint2d1(vector<cv::Point2f> point){
    vector<cv::Point2d> point_;
    for(int i=0;i<point.size();i++){
        point_.push_back(cv::Point2d((double)point[i].x,(double)point[i].y));
    }
    return point_;
}

// ORB ====================================================================================
/*
    IMG_2 :
    Keypoint_Ref:
    Descriptor_Ref :

*/
void ORB_featurematching(Mat& descriptors, vector<cv::KeyPoint>& keypoints, 
        Mat& img_1, Mat& img_2, Mat& left_pose, Mat& right_pose, Mat& Kd, Mat& Kd1,
        vector<Point2f>& ORB_features_temp, Mat& point3d_homo, double focal, cv::Point2d pp){

    right_pose = Mat::eye(4, 4, CV_64FC1); //next pose
    left_pose.copyTo(right_pose.rowRange(0,3).colRange(0,4));

    // 1. right image setting
    Ptr<ORB> detector=ORB::create();

    vector<cv::KeyPoint> TargetKeypoints;
    vector<cv::KeyPoint> Temp_Keypoints;
    vector<cv::KeyPoint> Temp1_Keypoints;


    cv::Mat TargetDescriptor;
    cv::Mat temp_TargetDescriptor;
    cv::Ptr<cv::DescriptorMatcher> Matcher_ORB=cv::BFMatcher::create(cv::NORM_HAMMING);

    detector->detectAndCompute(img_2, cv::Mat(), TargetKeypoints, TargetDescriptor);
    // 2. ORB Features Matching
    vector<cv::DMatch> matches;
    Matcher_ORB->match(descriptors, TargetDescriptor, matches);

    // 3. remain only good matches
    vector<cv::DMatch> good_matches;
    vector<Point2f> points1_ess;
    vector<Point2f> points2_ess;

    printf(" matched points size : %d\n",points1_ess.size());

    // 4. Epipolar Constraints 
    double a1=(-386.1448);
    double a2=0.0f;
    double a3=0.0f;
    /* Epipolar Constraints :: E=[t]_{x}R */ 
    double t[]={0.0f, -a3, a2,
                a3, 0.0f, -a1,
                -a2, a1, 0.0f};

    Mat skewt(3,3,CV_64FC1,t);
    Mat R=Mat::eye(3,3,CV_64FC1);
    Mat E1=skewt*R;

    vector<Point2f> points_L_temp;
    vector<Point2f> points_R_temp;

    /* Epipolar Constraints */  

    for (int i=0;i<matches.size();i++){
        points1_ess.push_back(keypoints[matches[i].queryIdx].pt);
        points2_ess.push_back(TargetKeypoints[matches[i].trainIdx].pt);
        Temp_Keypoints.push_back(keypoints[matches[i].queryIdx]);
    }

    for(int i=0;i<points1_ess.size();i++){
        double xx1[]={points1_ess[i].x,points2_ess[i].y,1};
        Mat x1t(1,3,CV_64FC1,xx1);
        Mat tmpmat=x1t*E1;
        double value=(tmpmat.at<double>(0,0)*points2_ess[i].x+
        tmpmat.at<double>(0,1)*points2_ess[i].y+
        tmpmat.at<double>(0,2)*1);

        float value1=abs(points1_ess[i].y-points2_ess[i].y);

        if(value1<1){
            points_L_temp.push_back(cv::Point2f(points1_ess[i].x, points1_ess[i].y));
            points_R_temp.push_back(cv::Point2f(points2_ess[i].x, points2_ess[i].y));   
            Temp1_Keypoints.push_back(Temp_Keypoints[i]);         
        }
    } 

    points1_ess=points_L_temp;
    points2_ess=points_R_temp;
    printf(" after Epipolar Constraints matched points size : (%d,%d)\n",points1_ess.size(),Temp1_Keypoints.size());


    vector<Point2d> points_L_temp_double=toPoint2d1(points1_ess);
    vector<Point2d> points_R_temp_double=toPoint2d1(points2_ess);

    // 4. Create 3D map points
    triangulatePoints(Kd*left_pose, Kd1*right_pose, points_L_temp_double, points_R_temp_double, point3d_homo);
    point3d_homo = convert_14to13(point3d_homo);
    ORB_features_temp=points1_ess;
    keypoints=Temp1_Keypoints;

}